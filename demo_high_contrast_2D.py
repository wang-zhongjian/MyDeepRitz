# coding=utf-8
"""High Contrast Problem"""
import time
import tensorflow as tf
import numpy as np
from tensorflow import random_normal_initializer as norm_init


class SolvePDEviaDeepRitz(object):
    """The neural network model."""
    def __init__(self, sess):
        self.sess = sess
        self.dimension=2
        self.r0=np.pi/6.28
        self.L=1.0
        self.alpha0=100.0
        self.alpha1=1.0
        self.n_layer = 5
        self.n_neuron = [self.dimension, 15, 15, 15, 15, 1]
        self.n_displaystep = 500
        self.batch_sizeE = 1024
        self.valid_size = 256
        self._extra_train_ops = []
        self.global_step = \
            tf.get_variable('global_step', [],
                            initializer=tf.constant_initializer(1),
                            trainable=False, dtype=tf.int32)
        self.X1 = tf.placeholder(tf.float64,
                                    [None, 1],
                                    name='X1')
        self.X2 = tf.placeholder(tf.float64,
                                    [None, 1],
                                    name='X2')
        self.Xedge1=tf.placeholder(tf.float64,[None,1],
                                   name='boundaryX1')
        self.Xedge2=tf.placeholder(tf.float64,[None,1],
                                   name='boundaryX2')
        self.gdata=tf.placeholder(tf.float64,[None,1],name='g_data')
        self.a=tf.placeholder(tf.float64,[None,1])
        self.f=tf.placeholder(tf.float64,[None,1])
        self.is_training = tf.placeholder(tf.bool)
        self.U=self._solving_net(self.X1,self.X2)
        self.Uedge=self._solving_net(self.Xedge1,self.Xedge2)
        self.lag_history = []
        self.err_history = []
        '''data world'''
        self.Xdata1=tf.placeholder(tf.float64,[None,1],
                                   name='dataX1')
        self.Xdata2=tf.placeholder(tf.float64,[None,1],
                                   name='dataX2')
        self.valdata=tf.placeholder(tf.float64,[None,1],
                                    name='dataValue')
        self.xx128,self.yy128=np.meshgrid(np.linspace(-self.L,self.L,num=self.valid_size+1),np.linspace(-self.L,self.L,num=self.valid_size+1)) # x[-1]=0,x[Nx]=L
        self.valid_dict=self.generate_valid_data()

    def build(self):
        start_time = time.time()
        self.bdd_cons = tf.reduce_mean(tf.square(self.Uedge))
        self.U1=tf.reshape(tf.gradients(self.U,self.X1),shape=tf.shape(self.X1))
        self.U2=tf.reshape(tf.gradients(self.U,self.X2),shape=tf.shape(self.X1))
        self.g1 = tf.reshape(tf.gradients(self.g, self.X1), shape=tf.shape(self.X1))
        self.g2 = tf.reshape(tf.gradients(self.g, self.X2), shape=tf.shape(self.X1))
        self.g=self.g_net(self.X1,self.X2)
        self.lagrangian_i = self.a * (tf.square(self.U1) + tf.square(self.U2)) / 2.0 - self.f * self.U+ self.a*(self.U1*self.g1+self.U2*self.g2)
        self.lagrangian=tf.reduce_mean(self.lagrangian_i)
        self.ref_U_grid=self.ref_generator()
        self.error_i=self.U-self.ref_U_grid
        self.error=tf.reduce_mean(tf.square(self.error_i))
        self.t_bd = time.time()-start_time
        self.dataError=tf.reduce_mean(tf.square(self._solving_net(self.Xdata1,self.Xdata2)-self.valdata))
        print('Building time: %.1f s'%self.t_bd)
        return 500.0*self.bdd_cons+self.lagrangian

    def approxBoundary(self):
        self.gedge=self.g_net(self.Xedge1,self.Xedge2)
        self.g= self.g_net(self.X1, self.X2)
        self.Lg=(-1)*(tf.reshape(tf.gradients(self.a*tf.reshape(tf.gradients(self.g,self.X1),shape=tf.shape(self.X1)),self.X1),shape=tf.shape(self.X1))
                + tf.reshape(tf.gradients(self.a * tf.reshape(tf.gradients(self.g, self.X2), shape=tf.shape(self.X1)),self.X2), shape=tf.shape(self.X1)))
        return tf.reduce_sum(tf.square(self.gedge-self.gdata))


    def generateX(self,nx):
        xsample=np.random.uniform(-self.L, self.L , size=[nx, self.dimension])
        return xsample[:,0:1], xsample[:,1:]

    def generateXedge(self,nx):
        xsample=np.random.uniform(-self.L,self.L,[nx,2])
        quanx=int(nx/4)
        for i in range(quanx):
            xsample[i,0]=-self.L
            xsample[i+quanx,0]=self.L
            xsample[i+2*quanx,1]=-self.L
            xsample[i+3*quanx,1]=self.L
        return xsample[:,0:1], xsample[:,1:]

    def ref_generator(self):
        ref_sol=np.zeros([self.valid_size+1,self.valid_size+1])
        for i in range(self.valid_size+1):
            for j in range(self.valid_size+1):
                r=np.sqrt(self.xx128[i,j]**2+self.yy128[i,j]**2)
                if r>self.r0:
                    ref_sol[i,j]=r**3/self.alpha0+(1/self.alpha1-1/self.alpha0)*self.r0**3
                else:
                    ref_sol[i, j] = r ** 3 / self.alpha1
        return np.reshape(ref_sol,[(self.valid_size+1)*(self.valid_size+1),1])

    def train(self,Loss_Function,n_maxstep,learning_rate,batch_size,is_first_training=False,is_training_g=False,is_second_training=False):
        start_time = time.time()
        # train operations
        if is_training_g:
            train_var=tf.trainable_variables(scope='g')
        else:
            train_var = tf.trainable_variables(scope='uhomo')
        grads = tf.gradients(Loss_Function, train_var)
        if is_first_training:
            optimizer = tf.train.AdamOptimizer(learning_rate,name='first_optimizer')
        else:
            if is_second_training:
                optimizer = tf.train.AdamOptimizer(learning_rate,name='second_optimizer')
        apply_op = \
            optimizer.apply_gradients(zip(grads, train_var),
                                      global_step=self.global_step)
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        if is_first_training:
            self.sess.run(tf.global_variables_initializer())
        else:
            uninitialized_vars = []
            for var in tf.global_variables():
                try:
                    self.sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
            self.sess.run(tf.variables_initializer(uninitialized_vars))
        '''Initializing'''
        # generate training data for the first time
        train_dict=self.generate_training_data(batch_size)
        # begin sgd iteration
        for _ in range(n_maxstep+1):
            step = self.sess.run(self.global_step)
            self.sess.run(self.train_op, # train operation
                          feed_dict=train_dict)
            if step % self.n_displaystep == 0 and is_training_g==False:
                sol,temp_lag,g=\
                    self.sess.run([self.U,self.lagrangian,self.g],feed_dict=self.valid_dict)
                temp_err=np.sqrt(np.mean(np.square(sol+g-self.ref_U_grid))/np.mean(np.square(self.ref_U_grid)))
                self.err_history.append(temp_err)
                self.lag_history.append(temp_lag)
                print("step: %5u, err: %.3e, lag: %.3e"
                      %(step, temp_err, temp_lag) +
                      " runtime: %4u" %
                      (time.time()-start_time+self.t_bd))
            if step % self.n_displaystep == 0 and is_training_g:
                g_err,gval=\
                    self.sess.run([Loss_Function,self.g],feed_dict=self.valid_dict)
                print("step: %5u, err: %.3e"
                      %(step, g_err) )

            if step % 10 == 0:
                train_dict=self.generate_training_data(batch_size= batch_size,batch_sizeE=self.batch_sizeE)
            step += 1
        end_time = time.time()
        print("running time: %.3f s" % (end_time-start_time))

    def generate_training_data(self,batch_size,batch_sizeE=1024):
        x1,x2=self.generateX(batch_size)
        x1edge,x2edge=self.generateXedge(batch_sizeE)
        f=np.zeros([batch_size,1])
        a=np.ones([batch_size,1])*self.alpha0
        for i in range(batch_size):
            r=np.sqrt(x1[i,0]**2+x2[i,0]**2)
            if r<self.r0:
                a[i,0]=self.alpha1
            f[i,0]=-9*r
        gdata=np.zeros([batch_sizeE,1])
        for i in range(batch_sizeE):
            r = np.sqrt(x1edge[i, 0] ** 2 + x2edge[i, 0] ** 2)
            gdata[i,0]=r**3/self.alpha0+(1/self.alpha1-1/self.alpha0)*self.r0**3
        return {self.X1:x1,self.X2:x2,self.Xedge1:x1edge,self.Xedge2:x2edge,self.a:a,self.f:f,self.gdata:gdata,self.is_training: True}

    def generate_valid_data(self,batch_sizeE=1024):
        batch_size = (self.valid_size+1) * (self.valid_size+1)
        x1=np.reshape(self.xx128,[batch_size,1])
        x2=np.reshape(self.yy128,[batch_size,1])
        x1edge, x2edge = self.generateXedge(batch_sizeE)
        f = np.zeros([batch_size, 1])
        a = np.ones([batch_size, 1]) * self.alpha0
        for i in range(batch_size):
            r = np.sqrt(x1[i, 0] ** 2 + x2[i, 0] ** 2)
            if r <= self.r0:
                a[i, 0] = self.alpha1
            f[i, 0] =- 9 * r
        gdata = np.zeros([batch_sizeE, 1])
        for i in range(batch_sizeE):
            r = np.sqrt(x1edge[i, 0] ** 2 + x2edge[i, 0] ** 2)
            gdata[i, 0] = r ** 3 / self.alpha0 + (1 / self.alpha1 - 1 / self.alpha0) * self.r0 ** 3
        return {self.X1: x1, self.X2: x2, self.Xedge1:x1edge,self.Xedge2:x2edge, self.a: a, self.f: f,self.gdata:gdata,
                self.is_training: False}

    def _solving_net(self, x1, x2):
        with tf.variable_scope('uhomo',reuse=tf.AUTO_REUSE):
            layer1 = self._one_layer(tf.concat([x1,x2],1),  self.n_neuron[1],std=10.0,
                                         name='layer1')
            layer2 = self._one_layer(tf.concat([x1,x2,layer1],1), self.n_neuron[2],std=10.0,
                                         name='layer2')
            layer3 = self._one_layer(tf.concat([layer1,layer2],1), self.n_neuron[3],std=10.0,
                                     name='layer3')
            layer4 = self._one_layer(tf.concat([layer2,layer3],1), self.n_neuron[4],std=10.0,
                                     name='layer4')
            u = self._one_layer(layer4, 1,std=5.0,
                                activation_fn=None, name='final')
        return u

    def g_net(self, x1, x2):
        with tf.variable_scope('g',reuse=tf.AUTO_REUSE):
            layer1 = self._one_layer(tf.concat([x1,x2],1),  10,
                                         name='layer1')
            layer2 = self._one_layer(tf.concat([x1,x2,layer1],1), 10,
                                         name='layer2')
            layer3 = self._one_layer(tf.concat([layer2],1), 10,
                                   name='layer3')
            g = self._one_layer(layer3, 1,
                                activation_fn=None, name='final')
        return g

    def _one_layer(self, input_, out_sz,
                   activation_fn=tf.nn.sigmoid,
                   std=1.0, name='linear'):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            shape = input_.get_shape().as_list()
            w = tf.get_variable('Weights',
                                [shape[1], out_sz], tf.float64,
                                norm_init(stddev=std/np.sqrt(shape[1]+out_sz)))
            b = tf.get_variable('Bias',
                                [1,out_sz], tf.float64,
                                norm_init(stddev=std/np.sqrt(1+out_sz)))
            affine = tf.matmul(input_, w)+b
        if activation_fn is not None:
            return activation_fn(affine)
        else:
            return affine




def main():
    tf.reset_default_graph()
    with tf.Session() as sess:
        print("Begin to solve PDE")
        model = SolvePDEviaDeepRitz(sess)
        loss_function=model.approxBoundary()
        model.train(Loss_Function=loss_function, n_maxstep=3000, learning_rate=5e-3, batch_size=1024,
                    is_first_training=True,is_training_g=True)
        loss_function = model.build()  # building model
        model.train(Loss_Function=loss_function, n_maxstep=5000, learning_rate=1e-3, batch_size=4096,
                    is_second_training=True)


if __name__ == '__main__':
    main()
