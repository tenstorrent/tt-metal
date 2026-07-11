# SDPA-owned perf tests: migrate to the real-time device program profiler

Branch: `skrstic/sdpa-perf-realtime-migration`

## Goal

Move the existing op-level perf tests for the SDPA-team-owned ops onto the
lightweight real-time device program profiler (`profile_realtime_program`,
`tests/ttnn/profiling/realtime_profiler_utils.py`), the same mechanism the
ring-joint SDPA perf checks already use (PR #48199). Previously these tests
measured device kernel duration through the Tracy device profiler — either an
out-of-process `run_device_profiler` subprocess (indexer_score) or an external
`python -m tracy` wrapper (the sparse perf smokes). Then verify the migrated
tests report the same numbers as `main`.

`topk_large_indices` is intentionally excluded: it has no perf test to migrate.

## What changed

Three test files, no new test cases (same functions, same parametrization) —
only the measurement mechanism changed.

### `tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py`
- `test_indexer_score_math_util` (5 cases: glm5, dsv32, minimax_m3, dsv32_tp1,
  glm5_tp1) now runs the op inline under `profile_realtime_program`
  (`collect_all=True`), takes the **min** device duration across the 5 dispatches
  (mirroring the old Tracy min-of-`DEVICE KERNEL DURATION`), and asserts the same
  `math_util` within the same ±2% band. The Tracy subprocess (`run_device_profiler`
  + `post_process_ops_log`) is gone.
- The RT record carries no core count, so the deployed effective core count
  (all these shapes fill 110 cores) is baked into each `_MATH_UTIL_CASES` entry,
  which now carries `(case_id, run_builder, mm_flops_thunk, expected_util,
  expected_cores)`.
- Added three run-builders (`_sp7_dsa_run`, `_short_seq_dsa_run`, `_m3_msa_run`)
  that stage inputs on device and return a 5×-dispatch thunk.
- The Tracy `*_perf_impl` helper tests are kept (skipif-CI) for manual
  `python -m tracy` inspection.
- Removed the now-unused `from unittest import mock` import.

### `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa.py`
- `test_sparse_sdpa_perf` (7 shapes) wraps the existing `run_op(...)` in a
  `profile_realtime_program` run and logs the in-test device-kernel duration.
  `SparseSDPAOperation` is the only device program dispatched (`records=1`), so
  the op's duration is isolated cleanly. Shape assert unchanged.

### `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa_msa_perf.py`
- `test_msa_perf_prod` and `test_msa_perf_prod_single_chip_gqa` (bf16 + bfp8)
  measure `run_op_msa_native(...)` via a shared `_profile_msa_native_duration_ns`
  helper and log the duration. Shape assert unchanged.

All three files hard-fail if `IsProgramRealtimeProfilerActive()` is false, i.e.
they require a Tracy-enabled build run with `TT_METAL_DEVICE_PROFILER=1`.

## Validation (8× Blackhole p150b, Tracy-enabled build)

Numbers match `main` within run-to-run measurement noise — the RT profiler
measures the same device kernel time Tracy does.

### indexer_score (`test_indexer_score_math_util`) — PASS in-band both ways

| case       | Tracy (main)        | RT profiler         | Δ util |
|------------|---------------------|---------------------|--------|
| glm5       | ~70.1% (pass)       | 69.94% (0.345 ms)   | ✓      |
| dsv32      | 75.99% (0.635 ms)   | 75.92% (0.636 ms)   | -0.07  |
| minimax_m3 | 43.10% (0.069 ms)   | 42.95% (0.069 ms)   | -0.15  |
| dsv32_tp1  | 77.31% (0.627 ms)   | 77.27% (0.628 ms)   | -0.04  |
| glm5_tp1   | 75.53% (0.321 ms)   | 75.40% (0.322 ms)   | -0.13  |

### sparse_sdpa / sparse_sdpa_msa — device kernel duration

| case                    | Tracy (main) | RT profiler | Δ       |
|-------------------------|--------------|-------------|---------|
| sparse_sdpa prod-dense  | 3316.2 µs    | 3325.2 µs   | +0.27%  |
| sparse_sdpa_msa bf16    | 1390.0 µs    | 1407.3 µs   | +1.24%  |
| sparse_sdpa_msa bfp8    | 777.0 µs     | 773.9 µs    | -0.41%  |

## How to run

```bash
# indexer_score band checks (env-gated)
TT_METAL_DEVICE_PROFILER=1 INDEXER_SCORE_PERF_CHECKS=1 \
  scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py::test_indexer_score_math_util

# sparse perf smokes
TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa.py::test_sparse_sdpa_perf
TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh --run-all \
  tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa_msa_perf.py
```
