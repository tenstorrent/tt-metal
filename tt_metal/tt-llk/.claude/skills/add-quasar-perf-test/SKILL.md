---
name: add-quasar-perf-test
description: Add Quasar LLK performance tests from existing correctness tests. Use when creating or extending perf tests for Quasar kernels, especially files under tests/python_tests/quasar or tests/sources/quasar.
user_invocable: true
---

# /add-quasar-perf-test - Add Quasar Perf Test

## Usage

```
/add-quasar-perf-test matmul
/add-quasar-perf-test eltwise_binary
/add-quasar-perf-test <op> --source tests/sources/quasar/<op>_quasar_test.cpp
```

## Goal

Create a Quasar perf test that reuses the correctness test harness, emits perf reports through `PerfConfig`, and gives each `PerfRunType` a valid single-kernel path in the C++ source.

## What to Do

1. Find the existing correctness test in `tests/python_tests/quasar/test_<op>_quasar.py` and the source in `tests/sources/quasar/<op>_quasar_test.cpp`.
2. Move reusable parameter axes from inline lambdas into named helpers when the perf test needs them, as `test_matmul_quasar.py` does for:
   - format, math fidelity, dest sync, dest accumulation, dimensions
   - implied math format, register format hint, direct indexing
3. Add a perf test file named `tests/python_tests/quasar/perf_<op>_quasar.py`.
4. Mark the perf test with both `@pytest.mark.perf` and `@pytest.mark.quasar`.
5. Import the correctness test as a helper, usually `from quasar.test_<op>_quasar import test_<op> as run_<op>`.
6. Parametrize the perf test with a narrow, perf-oriented sweep:
   - `run_types=PERF_RUN_TYPES` from `helpers.llk_params` (one `PerfRunType` per pytest case: L1_TO_L1, UNPACK_ISOLATE, MATH_ISOLATE, PACK_ISOLATE, L1_CONGESTION)
   - `loop_factor=[32]` unless the operation has an established different loop count
   - `is_perf=[True]`
   - prefer exact destination fill when dimensions affect profiler normalization
7. Thread `perf_report` into the helper and call it with `is_perf=True`.
8. In the correctness helper, add keyword-only `is_perf=False` and `perf_report=None`. When `is_perf` is true:
   - require `perf_report`
   - build `PerfConfig(run_types=run_types, **test_config_kwargs)`
   - call `configuration.run(perf_report)`
   - return before golden comparison
9. Preserve the normal correctness path with `TestConfig` and explicit `PERF_RUN_TYPE(PerfRunType.L1_TO_L1)` so the shared C++ source sees a defined perf type.

## C++ Source Requirements

For `tests/sources/quasar/<op>_quasar_test.cpp`:

- Include `perf.h` and `profiler.h`.
- Put setup in `ZONE_SCOPED("INIT")` and steady-state work in `ZONE_SCOPED("TILE_LOOP")`.
- Read `LOOP_FACTOR` from runtime parameters and use it in the tile loop.
- Branch by `PERF_RUN_TYPE` inside each TRISC section.
- In inactive isolate paths, either no-op or call the matching perf mock so the other active thread can complete handshakes.
- For `PACK_ISOLATE` and `L1_CONGESTION`, drain the packer each iteration when pack emits work:
  ` _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();`
- Keep `PROFILER_SYNC()` at the end of each profiled zone.

Use the current matmul branch as the reference implementation:

- `tests/python_tests/quasar/perf_matmul_quasar.py`
- `tests/python_tests/quasar/test_matmul_quasar.py`
- `tests/sources/quasar/matmul_quasar_test.cpp`

## Validation

- Run perf tests alone; do not combine them with other LLK tests.
- Use `/run-test` rather than direct `pytest`.
- First compile a focused perf variant, then run or rerun it.
- If a perf run reports negative or zero tile-loop cycles, inspect missing drains, missing `PROFILER_SYNC()`, or a thread leaving `TILE_LOOP` before its coprocessor finishes.
