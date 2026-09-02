# High Power Matmul Workload

*Read this whole doc before running anything. It's quite small so it won't take much time!*

Sustained HiFi4 matmul across all cores for power draw measurement.

C++ Test: `tt_metal/programming_examples/high_power_matmul/high_power_matmul.cpp`
Compute kernel: `tt_metal/programming_examples/high_power_matmul/kernels/compute/mm_power.cpp`
Data Movement kernels: `tt_metal/programming_examples/high_power_matmul/kernels/dataflow`

## Build

```bash
./build_metal.sh --build-programming-examples
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
```

*Note*: You don't need to rebuild if you're changing the kernels, they're JIT compiled. You do need to recompile if you're changing the C++ test.

You can put printing statements inside the kernels to debug/instrument the code, see example below. Please note that they add a small execution overhead, although since we're not debugging race conditions (yet), this might not matter. For more details ask Deepwiki or look at the tt-metal documentation.

## Run

To run the test with predefined values:
```bash
./build/programming_examples/metal_example_high_power_matmul
```

*Defaults*: 4096×4096×2048 (datums, not tiles), HiFi4, 500 iterations.

## Custom parameters

```bash
./build/programming_examples/metal_example_high_power_matmul [M] [N] [K] [iterations]
```

Examples:
```bash
# Longer, more compute-bound run
./build/programming_examples/metal_example_high_power_matmul 4096 4096 4096 1000

# Quick sanity check
./build/programming_examples/metal_example_high_power_matmul 2048 2048 2048 100

# If you wish to print out the iteration progress:
TT_METAL_DPRINT_CORES=0,0 ./build/programming_examples/metal_example_high_power_matmul 2048 2048 2048 500
```

All dimensions must be multiples of 32 (tile size)!

## Tuning for more power

- **Larger K** → more compute per output tile (more compute-bound)
- **More iterations** → longer sustained power draw
- **Larger M×N** → more output tiles across cores
