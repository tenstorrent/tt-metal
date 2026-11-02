.. _MatMul_Multi_Core_example:

Matmul (Multi Core Optimized)
=============================

The Tensix core architecture's secret weapon is its full user control over memory workload spread, core communication style, novel block matmul kernels, and compute patterns. In this section, we will harness real power through 3 shiny optimizations, each building off one another: data reuse, data multicast, and multidimensional systolic arrays (coming soon).

::

    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_matmul_multicore_reuse
    ./build/programming_examples/metal_example_matmul_multicore_reuse_mcast

.. toctree::

   matmul_multi_core_optimizations/data_reuse
   matmul_multi_core_optimizations/data_mcast
