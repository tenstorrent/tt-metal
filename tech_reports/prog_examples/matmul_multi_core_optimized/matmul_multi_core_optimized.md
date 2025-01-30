# Matmul (Multi Core Optimized)

The Tensix core architecture's secret weapon is its full user control over memory workload spread, core communication style, novel block matmul kernels, and compute patterns. In this section, we will harness real power through 3 shiny optimizations, each building off one another: data reuse, data multicast, and multidimensional systolic arrays (coming soon).

To build and execute, you may use the following commands:
```bash
    export TT_METAL_HOME=$(pwd)
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/matmul_multi_core_reuse
    ./build/programming_examples/matmul_multi_core_reuse_mcast
```

- [Matmul Multi_core Reuse (Optimized)](./data_reuse.md)
- [Matmul Multi_core Multi-Cast (Optimized)](./data_mcast.md)
