# minimal_matmul GEMM benchmarks

Manual-only pytest benchmarks for `ttnn.experimental.minimal_matmul` on a single Blackhole chip, built to fill
the partner GEMM spec sheet (theoretical peak, measured TFLOP/s, utilization per shape and data type).
All of them skip unless `TTNN_RUN_GEMM_FLOPS_BENCHMARK=1` is set, so they never run in CI by accident.
(`test_benchmark.py` in this directory is the unrelated `ttnn.matmul` GEMM FLOPS benchmark from
`tech_reports/GEMM_FLOPS`.)

| File | Role |
|---|---|
| `minimal_matmul_gemm_common.py` | Shared definitions: sheet shapes, dtype-column map, peak formula, measurement loop, metrics, PCC check |
| `test_minimal_matmul_block_sweep.py` | Phase 1: sweep `MinimalMatmulConfig` blockings per sheet cell, emit the best-blocking map. Also the compute-bound peak search. |
| `test_minimal_matmul_benchmark.py` | Phase 2: run every sheet cell with its hard-coded best blocking and write the sheet CSV |

All commands run from the repo root with the ttnn Python environment active and `TT_METAL_HOME` set to the
repo; every output lands under `$TT_METAL_HOME/generated/`. Add `--timeout 0` if your pytest-timeout ignores
the per-test marker (the repo default is 300 s and one sweep item takes longer).

## Data-type columns

| Sheet column | ttnn dtype | fidelity (passes) | dest accumulation |
|---|---|---|---|
| BF16 | `bfloat16` | HiFi2 (2) | bf16 |
| FP16 | none | | reported N/A: tt-metal has no FLOAT16 tensor dtype |
| TF32 / FP32 | `float32` | HiFi4 (4) | fp32 (one measurement, reported in both columns) |
| FP8 | `bfloat8_b` | LoFi (1) | bf16 |
| FP4 | `bfloat4_b` | LoFi (1) | bf16 |

Theoretical peak: `num_cores * 1.35e9 * (8*16*16) * 2 / passes / 1e12` TFLOP/s (Blackhole clock); on the
12x10 grid that is 332 / 166 / 664 for 2 / 4 / 1 passes. Utilization is measured over theoretical.

## 1. Block sweep (phase 1)

Sweeps M/K/N block sizes {1,2,4,8,16} tiles and the maximal-area sub-blocks per cell, timing each config as
the fastest of three executions of a captured trace. Appends to `generated/minimal_matmul_block_sweep.csv`
and rewrites `generated/minimal_matmul_best_blocking.json` plus a `BEST_BLOCKING = {...}` literal in the log.

```bash
# whole sheet, one data type per invocation, 16 chips in parallel (~15 min each on a Galaxy)
TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 TTNN_MINIMAL_MATMUL_NUM_CHIPS=16 TTNN_MINIMAL_MATMUL_SWEEP_RESET=1 \
  pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_block_sweep.py -k "block_sweep and BF16"
TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 TTNN_MINIMAL_MATMUL_NUM_CHIPS=16 \
  pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_block_sweep.py -k "block_sweep and FP32"
# ... likewise FP8, FP4 (they append to the same CSV)

# one cell on one chip
TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 \
  pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_block_sweep.py -k "block_sweep and BF16 and 8192x8192x8192"
```

Item ids are `dtype_column=<BF16|FP32|FP8|FP4>-<M>x<N>x<K>`. Environment knobs:

| Variable | Effect |
|---|---|
| `TTNN_MINIMAL_MATMUL_NUM_CHIPS=N` | Open the system mesh, carve 1x1 submeshes, deal configs round-robin to N chips (1..16). Default 1. |
| `TTNN_MINIMAL_MATMUL_SWEEP_RESET=1` | Start a fresh CSV. Default: append when the header matches, so a sweep can be split across invocations. |
| `TTNN_MINIMAL_MATMUL_BLOCK_SIZES=4,8` | Narrow the candidate block sizes. |
| `--grid-size 12x10` | Restrict the op to an x-by-y core grid. Default: the full compute grid. |

Then paste the logged `BEST_BLOCKING` literal over the one in `test_minimal_matmul_benchmark.py`.

## 2. Sheet benchmark (phase 2)

Runs every sheet cell, eager and trace, with its `BEST_BLOCKING` entry (op default for missing bf16 cells,
an explicit L1-safe fallback for missing fp32 cells) and writes `generated/minimal_matmul_gemm_sheet.csv`:
one row per cell and dispatch mode plus a PEAK row per column, with fidelity, accumulation, blocking,
theoretical and measured TFLOP/s, utilization, a 32-row PCC check and a note column. The CSV is complete
even if the test then fails on a PCC or unrunnable-blocking assertion.

```bash
TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_benchmark.py
# one dispatch mode, or a subset of rows (M x N x K)
... -k trace
TTNN_MINIMAL_MATMUL_SHAPES=8192x8192x8192,32x2048x2048 ...
```

About 7 minutes for the full sheet on one chip. The blockings are tuned for the 12x10 grid; pass
`--grid-size 12x10` on a chip with a larger grid to reproduce the sheet conditions, or resweep.

## 3. Peak GEMM: compute-bound shapes measured by math-kernel duration

The sheet shapes stream both operands from DRAM and are timed on the host, so they answer "time to result".
The peak GEMM row answers "what does the FPU sustain when fed", and two things change to get it:

1. **Shape.** `test_minimal_matmul_peak_search` sizes M and N so every core owns exactly one
   M_block x N_block output (8 or 16 tiles per axis on the grid in use) and sweeps K in {4096, 8192, 16384}
   with K_block and the sub-block. Every input tile then crosses DRAM once. K=16384 is enough to amortize
   pipeline fill; larger blocks only matter for FP8/FP4 (BF16 fits at most 8x16 or 16x8 per core in L1,
   FP32 only 8x8).
2. **Timing source.** With `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1` (single chip, build with
   `ENABLE_TRACY=ON`) every row also gets `device_time_ms`, `device_tflops`, `device_utilization_pct`: the
   average over the 30 traced op instances of the span from the first TRISC1 (math) kernel start on any core
   to the last TRISC1 kernel end on any core, converted at the clock the profiler reports. That excludes host
   dispatch, the gap between ops and the writer's output drain. It is the measurement behind the published
   `ttnn.matmul` utilization figures in `tech_reports/GEMM_FLOPS`, so it is the one to compare against them.
   The mid-run dump flag is required: without it the device log is only written at device close and the
   test fails looking for it.

```bash
# BF16 first with a fresh CSV, then the others append (TF32 shares the FP32 measurement)
TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TTNN_MINIMAL_MATMUL_SWEEP_RESET=1 \
  pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_block_sweep.py -k "peak_search and K16384 and BF16"
TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
  pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_block_sweep.py -k "peak_search and K16384 and (FP32 or FP8 or FP4)"
```

About 4 minutes for BF16 and 13 minutes for the other three on one chip. Rows land in
`generated/minimal_matmul_peak_gemm.csv`; the best rate per data type is logged at the end. The multi-chip
fan-out (`TTNN_MINIMAL_MATMUL_NUM_CHIPS`) does not apply here because the profiler log is per process.

Reference result on a Blackhole Galaxy chip, 12x10 grid, clock 1350 MHz under load (2026-09-15). "Host" is
the trace-timed rate of the same config, "TRISC1" the math-kernel rate; peaks are 332 / 166 / 664 TFLOP/s:

| column | shape (M x N x K) | blocking (Mb/Kb/Nb, sub) | host TFLOP/s | host util | TRISC1 TFLOP/s | TRISC1 util |
|---|---|---|---|---|---|---|
| BF16 (HiFi2) | 2560 x 6144 x 16384 | 8/4/16, 1x8 | 271.6 | 81.9% | 302.1 | 91.0% |
| TF32 / FP32 (HiFi4) | 2560 x 3072 x 16384 | 8/4/8, 1x4 | 137.2 | 82.7% | 154.2 | 92.9% |
| FP8 (LoFi) | 5120 x 6144 x 16384 | 16/4/16, 1x8 | 529.0 | 79.7% | 595.3 | 89.7% |
| FP4 (LoFi) | 5120 x 6144 x 16384 | 16/8/16, 1x8 | 564.4 | 85.0% | 603.0 | 90.9% |

For comparison, `tech_reports/GEMM_FLOPS` reports `ttnn.matmul` on a P150 (13x10) at 90.2% (BF16 HiFi2),
94.9% (BF16 HiFi4), 87.5% (BF8_B LoFi) and 90.5% (BF4_B LoFi) by the same TRISC1 measurement.

## Device-side utilization on the other tests

The same two profiler variables work on the block sweep and the sheet benchmark (single chip only) and add
the same three columns. Say which timing source a utilization number uses whenever you quote one: the two
differ by 8 to 10 points on these shapes.
