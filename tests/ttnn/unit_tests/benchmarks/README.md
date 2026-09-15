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

## 3. Compute-bound peak search

The sheet shapes stream both operands from DRAM. `test_minimal_matmul_peak_search` instead sizes M and N so
every core owns exactly one M_block x N_block output (8 or 16 tiles per axis on the grid in use), sweeps
K in {4096, 8192, 16384} with K_block and sub-block, and writes `generated/minimal_matmul_peak_gemm.csv`
with the best rate per data type logged at the end.

```bash
TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 TTNN_MINIMAL_MATMUL_NUM_CHIPS=16 \
  pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_block_sweep.py -k "peak_search and (BF16 or FP32)"
TTNN_RUN_GEMM_FLOPS_BENCHMARK=1 TTNN_MINIMAL_MATMUL_NUM_CHIPS=16 \
  pytest tests/ttnn/unit_tests/benchmarks/test_minimal_matmul_block_sweep.py -k "peak_search and (FP8 or FP4)"
```

## Device-side utilization

With a build that has `ENABLE_TRACY=ON`, set `TT_METAL_DEVICE_PROFILER=1` on any of the above (single chip
only) to add `device_time_ms`, `device_tflops` and `device_utilization_pct` columns computed from the
average TRISC1 (math) kernel duration, which excludes host dispatch.
