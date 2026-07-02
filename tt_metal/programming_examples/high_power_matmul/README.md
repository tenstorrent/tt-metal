# High Power Matmul Workload

*Read this whole doc before running anything. It's quite small so it won't take much time!*

Sustained HiFi4 matmul with configurable core count for power draw and performance measurement.
Sweep `num_cores` to produce a cores-vs-execution-time curve that can be crossed with power graphs.

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

*Defaults*: 4096×4096×4096 (datums, not tiles), HiFi4, 500 iterations, all cores.

## Custom parameters

```bash
./build/programming_examples/metal_example_high_power_matmul [M] [N] [K] [iterations] [num_cores]
```

`num_cores` is snapped down to the nearest whole number of rows in the compute grid (e.g. on
Blackhole with an 8-wide grid: 15 → 8, 20 → 16). Omit or pass 0 to use all cores.

Examples:
```bash
# Longer, more compute-bound run on all cores
./build/programming_examples/metal_example_high_power_matmul 4096 4096 4096 1000

# Quick sanity check
./build/programming_examples/metal_example_high_power_matmul 2048 2048 2048 100

# Run on exactly 8 cores (one row on Blackhole)
./build/programming_examples/metal_example_high_power_matmul 4096 4096 4096 500 8

# If you wish to print out the iteration progress:
TT_METAL_DPRINT_CORES=0,0 ./build/programming_examples/metal_example_high_power_matmul 2048 2048 2048 500
```

All dimensions must be multiples of 32 (tile size)!

## Cores-vs-time sweep (for crossing with power graphs)

Keep M, N, K, and iterations fixed, sweep `num_cores`. Example for Blackhole (13×10 = 130 cores):

```bash
BIN=./build/programming_examples/metal_example_high_power_matmul
for cores in 13 26 39 52 65 78 91 104 117 130; do
    echo -n "cores=$cores  "
    $BIN 4096 4096 4096 100 $cores 2>/dev/null | grep "Per-iteration"
done
```

`num_cores` snaps down to whole rows, so pass multiples of the grid width (13 on Blackhole) to
get exactly the requested count. The binary prints the actual grid used.

This produces the execution-time axis. Cross it against your power readings (sampled at the same
`num_cores` values) to find the efficiency knee.

## Tuning for more power

- **Larger K** → more compute per output tile (more compute-bound)
- **More iterations** → longer sustained power draw
- **Larger M×N** → more output tiles across cores
