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

## Power-experiment scenarios

To isolate *which* part of the pipeline actually costs power — NoC read, FPU compute, or NoC
write — rather than just keeping a core alive and cycling its circular-buffer handshake, any
kernel can be switched to "idle" mode. An idle kernel still performs its full CB handshake, so
the other kernels are stimulated exactly as before and none of them deadlock; it just skips its
real NoC transfer or FPU work.

These are read from the environment at startup and passed to the kernels as JIT defines, so
changing a scenario only recompiles the kernels — the host binary never needs rebuilding.

| Env var | Effect |
|---|---|
| `HIGH_POWER_DISABLE_READER=1` | Reader keeps `cb_reserve_back`/`cb_push_back` on both input CBs but skips `noc_async_read_tile` + barrier. Input tiles hold stale L1 data. |
| `HIGH_POWER_DISABLE_COMPUTE=1` | Compute keeps its CB and tile-register handshake but skips `matmul_tiles` + `pack_tile`. Output tiles hold garbage. |
| `HIGH_POWER_DISABLE_WRITER=1` | Writer keeps `cb_wait_front`/`cb_pop_front` but skips `noc_async_write_tile` + barrier. Output DRAM stays stale. |
| `HIGH_POWER_WRITE_AMPLIFICATION_PCT=<pct>` | The reader issues `2*Kt` NoC reads per output tile while the writer issues only 1. This re-writes each output tile `round((pct/100) * 2*Kt)` times (min 1) to load the write path symmetrically; `100` matches the reader's read volume. Unset/0 = normal. |

None of these need a correctness check: the workload never verifies its output, so stale or
garbage data is harmless. Do not use these modes for anything but power comparison.

### `POWER_CASE` — the five canonical scenarios

`POWER_CASE` overrides all four flags above at once. Cases 1–4 hold the writer at 100%
amplification so that turning the reader and/or compute off is a clean single-variable
comparison against case 1.

| `POWER_CASE` | Reader | Compute | Writer | Write amplification | Conventional subdir |
|---|---|---|---|---|---|
| `0` | real | real | real | off (baseline) | `regular` |
| `1` | real | real | real | 100% | `writer_amp` |
| `2` | real | **idle** | real | 100% | `compute_idle` |
| `3` | **idle** | **idle** | real | 100% | `reader_compute_idle` |
| `4` | **idle** | real | real | 100% | `reader_idle2` |

Note there is deliberately no writer-idle `POWER_CASE`; use
`HIGH_POWER_DISABLE_WRITER=1` directly for that comparison.

```bash
POWER_CASE=2 ./build/programming_examples/metal_example_high_power_matmul 4096 8192 8192 160
```

A confirmation line is printed at startup:

```
POWER_CASE=2 -- reader=real compute=idle writer=real write_amplification_pct=100
```

If `POWER_CASE` is unset the four individual flags are used as-is, for finer manual control.
