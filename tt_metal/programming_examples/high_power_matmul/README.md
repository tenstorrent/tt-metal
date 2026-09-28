# High Power Matmul Workload

*Read this whole doc before running anything. It's quite small so it won't take much time!*

Sustained HiFi4 matmul (or another per-tile op, see `HIGH_POWER_OP`) swept across a series of
core grids, for power-draw measurement. Every grid prints its wall-clock start/end time so an
external telemetry sampler can attribute power samples to it.

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

```bash
./build/programming_examples/metal_example_high_power_matmul [M] [N] [K] [iterations] [fixed_tiles_per_core]
```

*Defaults*: 256×256×512 (datums, not tiles), HiFi4, 100000 iterations, split mode.

| Positional arg | Meaning |
|---|---|
| `M N K` | matmul shape in datums; each must be a multiple of 32 (tile size) |
| `iterations` | how many times the whole `C = A x B` is repeated per grid |
| `fixed_tiles_per_core` | `0` (default) = **split mode**: the same total work is divided across the active cores, so every grid does identical FLOPs and pJ/FLOP is directly comparable between grids. `N > 0` = **fixed mode**: every core does exactly `N` output blocks, so total work grows with the grid. |

Examples:
```bash
# The shape used for the Wormhole-vs-Blackhole energy measurements
./build/programming_examples/metal_example_high_power_matmul 1024 2048 2048 160

# Quick sanity check
./build/programming_examples/metal_example_high_power_matmul 2048 2048 2048 100

# If you wish to print out the iteration progress:
TT_METAL_DPRINT_CORES=0,0 ./build/programming_examples/metal_example_high_power_matmul 2048 2048 2048 500
```

### The grid sweep

The program runs the workload once per core grid, pausing 5 s between grids so the idle floor
can be measured in between. The grid list is generated from the device's compute grid: for each
width `x` from 3 up to the device maximum, the heights `x-1`, `x` and `x+1` are tried and clipped
to the device. On a Wormhole n300 (8x7) that is 15 grids from 3x2 to 8x7; on a Wormhole n150
(8x8) 17 grids ending at 8x8; on a Blackhole p100a (11x10) 24 grids ending at 11x10. Pairs of
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
inputs, shape and op, the checksum must be identical for every grid and every block size.

## Tuning for more power

- **Larger K** → more compute per output tile (more compute-bound)
- **More iterations** → longer sustained power draw
- **Larger M×N** → more output tiles across cores
- **Output blocking** (below) → fewer DRAM reads per FLOP, so the FPUs rather than the NoC set the power

## Power-experiment knobs

All of these are read from the environment at startup and passed to the kernels as JIT defines,
so changing one only recompiles the kernels — the host binary never needs rebuilding. None of
them need a correctness check beyond the checksum: the workload never verifies its output, so
stale or garbage data is harmless. Do not use the idle modes for anything but power comparison.

### Idling a kernel

To isolate *which* part of the pipeline actually costs power — NoC read, FPU compute, or NoC
write — rather than just keeping a core alive and cycling its circular-buffer handshake, any
kernel can be switched to "idle" mode. An idle kernel still performs its full CB handshake, so
the other kernels are stimulated exactly as before and none of them deadlock; it just skips its
real NoC transfer or FPU work.

| Env var | Effect |
|---|---|
| `HIGH_POWER_DISABLE_READER=1` | Reader keeps `cb_reserve_back`/`cb_push_back` on both input CBs but skips `noc_async_read_tile` + barrier. Input tiles hold stale L1 data. |
| `HIGH_POWER_DISABLE_COMPUTE=1` | Compute keeps its CB and tile-register handshake but skips the math + `pack_tile`. Output tiles hold garbage. |
| `HIGH_POWER_DISABLE_WRITER=1` | Writer keeps `cb_wait_front`/`cb_pop_front` but skips `noc_async_write_tile` + barrier. Output DRAM stays stale. |
| `HIGH_POWER_WRITE_AMPLIFICATION_PCT=<pct>` | The reader issues `2*Kt` NoC reads per output tile while the writer issues only 1. This re-writes each output tile `round((pct/100) * 2*Kt)` times (min 1) to load the write path symmetrically; `100` matches the reader's read volume. Unset/0 = normal. |

### `POWER_CASE` — the six canonical scenarios

`POWER_CASE` overrides the four flags above at once. Cases 1–5 hold the writer at 100%
amplification so that turning exactly one of reader / compute / writer off is a clean
single-variable comparison against case 1.

| `POWER_CASE` | Reader | Compute | Writer | Write amplification | Conventional subdir |
|---|---|---|---|---|---|
| `0` | real | real | real | off (baseline) | `regular` |
| `1` | real | real | real | 100% | `writer_amp` |
| `2` | real | **idle** | real | 100% | `compute_idle` |
| `3` | **idle** | **idle** | real | 100% | `reader_compute_idle` |
| `4` | **idle** | real | real | 100% | `reader_idle2` |
| `5` | real | real | **idle** | 100% | `writer_idle` |

Case 0 was used as a stand-in for "writer idle" before case 5 existed, but it answers a
different question: turning amplification off measures the marginal cost of the extra writes,
not the writer's full contribution.

```bash
POWER_CASE=2 ./build/programming_examples/metal_example_high_power_matmul 1024 2048 2048 160
```

If `POWER_CASE` is unset the four individual flags are used as-is, for finer manual control.

### `HIGH_POWER_BLOCK_M` / `HIGH_POWER_BLOCK_N` — output blocking with tile reuse

By default (1x1) each core computes one output tile at a time and re-reads a full row of A and
column of B from DRAM for every one of them: 2 tile reads per multiply, an arithmetic intensity
of ~16 FLOP/byte, which makes the workload bandwidth-bound. With blocking, each core accumulates
a `BLOCK_M x BLOCK_N` patch of output tiles in the destination registers at once, so one column
slice of A and one row slice of B feed `BLOCK_M*BLOCK_N` multiplies. Tile reads per multiply
fall to `(BLOCK_M + BLOCK_N) / (BLOCK_M * BLOCK_N)`: at 2x4 that is 0.75, a 2.67x traffic
reduction, lifting intensity to ~42 FLOP/byte.

Constraints: `BLOCK_M * BLOCK_N <= 8` (the destination register budget), and `Mt`, `Nt` must be
divisible by `BLOCK_M`, `BLOCK_N`. In fixed mode `fixed_tiles_per_core` counts output *blocks*.

```bash
HIGH_POWER_BLOCK_M=2 HIGH_POWER_BLOCK_N=4 POWER_CASE=0 \
  ./build/programming_examples/metal_example_high_power_matmul 1024 2048 2048 160
```

The checksum must not change with the block size; if it does, the tile indexing is wrong.

### `HIGH_POWER_OP` — which instruction the compute kernel runs

| Value | Math per tile pair | Unit |
|---|---|---|
| `matmul` (default) | `matmul_tiles`, accumulating over K | FPU |
| `add` | `add_tiles` | FPU |
| `exp`, `gelu`, `recip`, `silu`, `sigmoid` | `copy_tile` from A, then the SFPU op in place | SFPU |

The reader and writer are identical for every value, so every op streams byte-identical DRAM
traffic through the same circular buffers at the same rate; the only thing that varies is what
the math unit does with each tile pair. The difference in dynamic energy between two ops is
therefore attributable to the instruction, not to data movement. The unary SFPU ops still wait
on and pop the second input CB so the reader's traffic and back-pressure are unchanged.

FLOP counts printed by the program assume matmul (`2*M*N*K` per iteration); for the other ops
they are a placeholder and the meaningful unit is energy per tile, or per byte moved.

```bash
HIGH_POWER_OP=exp POWER_CASE=1 ./build/programming_examples/metal_example_high_power_matmul 1024 2048 2048 160
```
