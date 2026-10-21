# Long Matmul Workload

*Read this whole doc before running anything. It's quite small so it won't take much time!*

A long-running HiFi4 matmul (or another per-tile op, see `LONG_MATMUL_OP`) swept across a
series of core grids. The point is not to stress the Tensix as hard as possible, but to keep each
run going long enough that tt-smi / ARC telemetry, which samples at tens of Hz, collects enough
samples per grid to give meaningful power numbers. Every grid prints its wall-clock start/end
time so an external telemetry sampler can attribute its samples to it.

C++ Test: `tt_metal/programming_examples/long_matmul/long_matmul.cpp`
Compute kernel: `tt_metal/programming_examples/long_matmul/kernels/compute/compute.cpp`
Data Movement kernels: `tt_metal/programming_examples/long_matmul/kernels/dataflow`

## Build

```bash
./build_metal.sh --build-programming-examples
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
```

*Note*: You don't need to rebuild if you're changing the kernels, they're JIT compiled. You do need to recompile if you're changing the C++ test.

You can put printing statements inside the kernels to debug/instrument the code, see example below. Please note that they add a small execution overhead, although since we're not debugging race conditions (yet), this might not matter. For more details ask Deepwiki or look at the tt-metal documentation.

## Run

```bash
./build/programming_examples/metal_example_long_matmul [M] [N] [K] [iterations] [fixed_blocks_per_core]
```

*Defaults*: 256×256×512 (datums, not tiles), HiFi4, 100000 iterations, split mode. With no
arguments at all it does a quick smoke run instead: the full grid only, 10 iterations, no pauses.

| Positional arg | Meaning |
|---|---|
| `M N K` | matmul shape in datums; each must be a multiple of 32 (tile size) |
| `iterations` | how many times the whole `C = A x B` is repeated per grid |
| `fixed_blocks_per_core` | `0` (default) = **split mode**: the same total work is divided across the active cores, so every grid does identical FLOPs and pJ/FLOP is directly comparable between grids. `N > 0` = **fixed mode**: every core does exactly `N` output blocks (one block is `LONG_MATMUL_BLOCK_M x LONG_MATMUL_BLOCK_N` tiles, 1x1 by default), so total work grows with the grid. |

Examples:
```bash
# The shape used for the Wormhole-vs-Blackhole energy measurements
./build/programming_examples/metal_example_long_matmul 1024 2048 2048 160

# Quick sanity check
./build/programming_examples/metal_example_long_matmul 2048 2048 2048 100

# If you wish to print out the iteration progress:
TT_METAL_DPRINT_CORES=0,0 ./build/programming_examples/metal_example_long_matmul 2048 2048 2048 500
```

### The grid sweep

The program runs the workload once per core grid, pausing 5 s between grids so the idle floor
can be measured in between. The grid list is generated from the device's compute grid: for each
width `x` from 3 up to the device maximum, the heights `x-1`, `x` and `x+1` are tried and clipped
to the device. On a Wormhole n300 (8x7) that is 15 grids from 3x2 to 8x7; on a Wormhole n150
(8x8) 17 grids ending at 8x8; on a Blackhole p100a (11x10) 24 grids ending at 11x10. Widths
past that band, where even `x-1` is taller than the device, run at the full height, so a 13x10
grid gets 26 grids ending 11x10, 12x10, 13x10. Pairs of
equal core count but different shape (3x4 vs 4x3, 6x7 vs 7x6) are included on purpose, to
separate a core-count effect from a NoC-shape one.

### Startup banner

Every effective knob is printed before the first grid, so the log of any run says exactly what
was measured. External tooling relies on the `POWER_CASE=` line staying in this format.

```
POWER_CASE=2 -- reader=real compute=idle writer=real write_amplification_pct=100 op=matmul
Output block: 2x4 tiles  |  256 blocks  |  0.750 DRAM tile reads per multiply (1x1 baseline = 2.000)
```

After each grid an `Output checksum: sum=... absmax=...` line is printed. The workload never
verifies its output, so this is the only signal that the tile indexing is right: with the same
inputs, shape and op, the checksum must be identical for every grid and every block size. This
holds in split mode only: in fixed mode the cores may not cover every output block, so blocks
nobody writes keep whatever DRAM held before.

## Tuning the workload

- **Larger K** → more compute per output tile (more compute-bound)
- **More iterations** → longer runs, so more telemetry samples per grid
- **Larger M×N** → more output tiles across cores
- **Output blocking** (below) → fewer DRAM reads per FLOP, so the FPUs rather than the NoC set the power

## Power-experiment knobs

All of these are read from the environment at startup and passed to the kernels as JIT defines.
The compute kernel is compiled with only the selected op, and an idle writer without its writes;
an idle reader or idle compute keeps its code, since in `updated` mode it still does its first two
steps for real. None of them need a correctness check beyond the checksum: the workload never verifies its output, so
stale or garbage data is harmless. Do not use the idle modes for anything but power comparison.

### Idling a kernel

To isolate *which* part of the pipeline actually costs power — NoC read, FPU compute, or NoC
write — rather than just keeping a core alive and cycling its circular-buffer handshake, any
kernel can be switched to "idle" mode. An idle kernel still performs its full CB handshake, so
the other kernels move the same data and none of them deadlock; it just skips its real NoC
transfer or FPU work.

Two side effects to keep in mind when comparing cases:

- **Data.** In the default `updated` mode an idle reader still reads its first two slices for
  real and idle compute still computes its first two blocks, so both halves of each
  double-buffered circular buffer hold real data, which the idle loop then recycles. In `legacy`
  mode (see below) compute works on whatever the input buffers already hold in L1 and the writer
  sends output buffers that were never packed: zeros or an earlier run's data, so operand values
  (and with them switching power) depend on what ran before.
- **Pacing.** Idling one kernel can change how the other two pace each other. Measured on n150
  at `1024 2048 2048`, 1x1, 8x8: compute idle (case 2) is 1% faster per iteration than case 1 in
  `legacy` mode and 5% faster in `updated` mode.

| Env var | Effect |
|---|---|
| `LONG_MATMUL_DISABLE_READER=1` | Reader keeps `cb_reserve_back`/`cb_push_back` on both input CBs but skips `noc_async_read_page` + barrier (after the first two slices, in `updated` mode). |
| `LONG_MATMUL_DISABLE_COMPUTE=1` | Compute keeps its CB and tile-register handshake but skips the math + `pack_tile` (after the first two blocks, in `updated` mode). |
| `LONG_MATMUL_DISABLE_WRITER=1` | Writer keeps `cb_wait_front`/`cb_pop_front` but skips `noc_async_write_page` + barrier. Output DRAM stays stale. |
| `LONG_MATMUL_WRITE_AMPLIFICATION_PCT=<pct>` | The reader issues `Kt * (BLOCK_M + BLOCK_N) / (BLOCK_M * BLOCK_N)` NoC reads per output tile (`2*Kt` at the default 1x1 block) while the writer issues only 1. This re-writes each output tile `round((pct/100) * reads)` times (min 1) to load the write path symmetrically; `100` matches the reader's read volume. Unset/0 = normal. |

### `LONG_MATMUL_MODE` — `updated` (default) or `legacy`

`legacy` reproduces how the published tt-ember results (#57900) were measured; use it to recreate
those figures. `updated` fixes the effects found since. With amplification off and no idle
kernel (`POWER_CASE=0`, the default flags) the two modes run the same kernels, apart from where
the clock starts.

| | `legacy` | `updated` |
|---|---|---|
| Amplified writes | every re-write waits for its own NoC barrier, so at 100% amplification the writer sets the pace | re-writes go out back to back, one barrier per tile |
| Amplification count with blocking | `2*Kt` per tile whatever the block | matches the blocked reader rate |
| Idle reader / idle compute data | whatever L1 already holds | the first two slices / blocks are real |
| Timed interval | starts before `EnqueueMeshWorkload`, so it includes any kernel JIT compile and possibly the tail of the input uploads | waits for the uploads to finish, then starts after `EnqueueMeshWorkload` returns |

Measured on Wormhole n150, `1024 2048 2048 20`, 8x8, per-iteration time:

| | case 0, 1x1 | case 1, 1x1 | case 0, 2x4 | case 1, 2x4 |
|---|---|---|---|---|
| `legacy` | 7.87 ms | 11.75 ms | 3.03 ms | 11.60 ms |
| `updated` | 7.79 ms | 11.90 ms | 2.95 ms | 4.28 ms |

In case 1 at 1x1 the writer is busy 94% of its lifetime in `legacy` and 56% in `updated`, so
`updated` no longer makes the writer look like the bottleneck. The iteration is just as slow in
both: the extra writes compete with the reads for DRAM and NoC bandwidth, so case 1 stays about
50% slower than case 0. With 2x4 blocking, `legacy` still writes `2*Kt` times per tile, 2.7x the
reader's volume, which is why its case 1 is almost 4x slower than case 0.

### `POWER_CASE` — the six canonical scenarios

`POWER_CASE` overrides the four flags above at once. Cases 1–5 hold the writer at 100%
amplification so that turning exactly one of reader / compute / writer off compares against
case 1 with the other two unchanged.

How much the writer's extra traffic slows the pipeline depends on `LONG_MATMUL_MODE` (above).

| `POWER_CASE` | Reader | Compute | Writer | Write amplification | Conventional subdir |
|---|---|---|---|---|---|
| `0` | real | real | real | off (baseline) | `regular` |
| `1` | real | real | real | 100% | `writer_amp` |
| `2` | real | **idle** | real | 100% | `compute_idle` |
| `3` | **idle** | **idle** | real | 100% | `reader_compute_idle` |
| `4` | **idle** | real | real | 100% | `reader_idle2` |
| `5` | real | real | **idle** | 100% | `writer_idle` |

Case 0 and case 5 answer different questions: case 0 turns write amplification off, so against
case 1 it measures the cost of the extra writes; case 5 idles the writer, so it measures the
writer's whole contribution.

```bash
POWER_CASE=2 ./build/programming_examples/metal_example_long_matmul 1024 2048 2048 160
```

If `POWER_CASE` is unset the four individual flags are used as-is, for finer manual control.

### `LONG_MATMUL_BLOCK_M` / `LONG_MATMUL_BLOCK_N` — output blocking with tile reuse

By default (1x1) each core computes one output tile at a time and re-reads a full row of A and
column of B from DRAM for every one of them: 2 tile reads per multiply, an arithmetic intensity
of ~16 FLOP/byte, which makes the workload bandwidth-bound. With blocking, each core accumulates
a `BLOCK_M x BLOCK_N` patch of output tiles in the destination registers at once, so one column
slice of A and one row slice of B feed `BLOCK_M*BLOCK_N` multiplies. Tile reads per multiply
fall to `(BLOCK_M + BLOCK_N) / (BLOCK_M * BLOCK_N)`: at 2x4 that is 0.75, a 2.67x traffic
reduction, lifting intensity to ~42 FLOP/byte.

Constraints: `BLOCK_M * BLOCK_N <= 8` (the destination register budget), and `Mt`, `Nt` must be
divisible by `BLOCK_M`, `BLOCK_N`. In fixed mode `fixed_blocks_per_core` counts these blocks, not tiles.

```bash
LONG_MATMUL_BLOCK_M=2 LONG_MATMUL_BLOCK_N=4 POWER_CASE=0 \
  ./build/programming_examples/metal_example_long_matmul 1024 2048 2048 160
```

The checksum must not change with the block size; if it does, the tile indexing is wrong.

### `LONG_MATMUL_OP` — which instruction the compute kernel runs

| Value | Math per tile pair | Unit |
|---|---|---|
| `matmul` (default) | `matmul_tiles`, accumulating over K | FPU |
| `add` | `add_tiles` | FPU |
| `exp`, `gelu`, `recip`, `silu`, `sigmoid` | `copy_tile` from A, then the SFPU op in place | SFPU |

The reader and writer are identical for every value, so every op streams byte-identical DRAM
traffic through the same circular buffers; the only thing that varies is what the math unit does
with each tile pair. The unary SFPU ops still wait on and pop the second input CB so the
reader's traffic is unchanged. The ops also run at the same rate as long as the reader sets the
pace: on n150 at `1024 2048 2048`, matmul, add and exp have the same reader time and
per-iteration time at 1x1 and 2x4. With a compute-bound shape the cheaper ops (add runs at LoFi,
the unary ones are a copy plus one SFPU op) would drain the buffers faster, and the energy
difference would no longer be the instruction alone.

FLOP counts printed by the program assume matmul (`2*M*N*K` per iteration); for the other ops
they are a placeholder and the meaningful unit is energy per tile, or per byte moved.

```bash
LONG_MATMUL_OP=exp POWER_CASE=1 ./build/programming_examples/metal_example_long_matmul 1024 2048 2048 160
```

## Profiling the kernels

With the device profiler on, each kernel records when it starts and ends and how its time splits
between waiting on circular buffers and doing real work. The tt-ember parser
(`tools/tt-ember/parser.py`, added in #58134) turns these into a per-core wait / active /
overhead breakdown for every grid.

```bash
TT_METAL_DEVICE_PROFILER=1 ./build/programming_examples/metal_example_long_matmul 1024 2048 2048 160
```

The markers land in `generated/profiler/.logs/profile_log_device.csv`, read back after every grid:

| Kernel | Markers | Wait = blocked in | Active = time in |
|---|---|---|---|
| Reader | `READER_KERNEL_START/END`, `READER_WAIT_CYCLES`, `READER_TRANSFER_CYCLES` | `cb_reserve_back` | `noc_async_read_page` + barrier |
| Compute | `COMPUTE_KERNEL_START/END`, `COMPUTE_WAIT_CYCLES`, `COMPUTE_COMPUTE_CYCLES` | `cb_wait_front` (TRISC0), `tile_regs_acquire` (TRISC1), `tile_regs_wait` + `cb_reserve_back` (TRISC2) | the math loop (TRISC1), `pack_tile` (TRISC2) |
| Writer | `WRITER_KERNEL_START/END`, `WRITER_WAIT_CYCLES`, `WRITER_TRANSFER_CYCLES` | `cb_wait_front` | `noc_async_write_page` + barrier |

Each kernel records these once per run, not per tile. The compute markers appear three times per
core, once per TRISC, since all three run the same code. The timestamp reads are compiled in only
when the profiler is on (see `kernels/profile_markers.h`), so a normal power-measurement run
executes the same instructions as a build without them.
