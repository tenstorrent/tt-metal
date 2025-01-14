# Matmul (Multi Core Optimized)

The Tensix core architecture's secret weapon is its full user control over memory workload spread, core communication style, novel block matmul kernels, and compute patterns. In this section, we will harness real power through 3 shiny optimizations, each building off one another: data reuse, data multicast, and multidimensional systolic arrays (coming soon).

Run the appropriate command for the Tenstorrent card you have installed:

| Card             | Command                              |
|------------------|--------------------------------------|
| Grayskull        | ```export ARCH_NAME=grayskull```     |
| Wormhole         | ```export ARCH_NAME=wormhole_b0```   |
| Blackhole        | ```export ARCH_NAME=blackhole```     |

Then run the following:
```bash
    export TT_METAL_HOME=$(pwd)
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/matmul_multi_core_reuse
    ./build/programming_examples/matmul_multi_core_reuse_mcast
```

- [Matmul Multi_core Reuse (Optimized)](./data_reuse.md)
- [Matmul Multi_core Multi-Cast (Optimized)](./data_mcast.md)
