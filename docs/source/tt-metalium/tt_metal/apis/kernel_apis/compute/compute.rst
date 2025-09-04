Compute APIs
============

This page contains a list of APIs avaliable in the compute kernels. These API govern differnt aspects of the compute kernel. Including

* Synchronization between the 3 cores runing the compute kernel cooperatively
* Perform the computation
* Copying data between SRAM and compute engines

Synchronization
---------------
.. toctree::

  acquire_dst
  release_dst
  reg_api

Register IO
-----------

.. toctree::

  cb_wait_front
  cb_pop_front
  cb_reserve_back
  cb_push_back

Compute
-------

Initialization
~~~~~~~~~~~~~~

.. toctree::

  init_functions
  binary_op_init_funcs

Compute (FPU/matrix engine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::

  add_tiles
  sub_tiles
  mul_tiles
  add_tiles_bcast
  sub_tiles_bcast
  mul_tiles_bcast

  matmul_tiles
  matmul_block
  reduce_tile
  transpose_wh_tile

  tilize
  untilize

.. only:: not html

Compute (SFPU/vector engine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Element wise operations
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  abs_tile
  exp_tile
  exp2_tile
  expm1_tile
  recip_tile
  sign_tile
  sqrt_tile
  rsqrt_tile
  log_tile
  log_with_base_tile
  power_tile
  rsub_tile
  signbit_tile
  square_tile
  ceil_tile
  clamp_tile
  cumsum_tile
  div_binary_tile
  add_binary_tile
  compute_kernel_hw_startup
  isinf_tile
  isnan_tile
  i0_tile
  logical_not_unary_tile
  heaviside_tile
  ltz_tile
  eqz_tile
  lez_tile
  gtz_tile
  gez_tile
  nez_tile
  unary_ne_tile
  unary_gt_tile
  unary_lt_tile
  unary_max_tile
  unary_min_tile
  gcd_tile
  add_int_tile
  asinh_tile
  atanh_tile
  binop_with_scalar_tile

Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::

  erf_tile
  erfc_tile
  erfinv_tile

  tanh_tile
  tan_tile
  sin_tile
  cos_tile
  asin_tile
  atan_tile
  acos_tile
  acosh_tile

Activation functions
^^^^^^^^^^^^^^^^^^^^

.. toctree::

  relu_tile
  elu_tile
  gelu_tile
  sigmoid_tile
  celu_tile

Misc
^^^^

.. toctree::

  quant_tile
  requant_tile
  dequant_tile
