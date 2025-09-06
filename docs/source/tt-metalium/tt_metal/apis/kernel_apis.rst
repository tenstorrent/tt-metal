Kernel APIs
===========

Kernel APIs are a set of functions that can be used inside the kernels. Some APIs are shared across all kernel types, while others are specific to certain kernel types (e.g., compute kernels). They are the primitives that enable the kernel to perform its operations, such as data movement, computation, and synchronization.

Common APIs
-----------

.. toctree::
  kernel_apis/circular_buffers/circular_buffers
  kernel_apis/kernel_args/kernel_args

Data Movement
-------------
.. toctree::

  kernel_apis/data_movement/data_movement

Compute
-------

.. toctree::
  kernel_apis/compute/compute
  kernel_apis/pack_unpack/packing_apis
  kernel_apis/sfpu/llk
