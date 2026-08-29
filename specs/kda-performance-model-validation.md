# KDA theoretical performance model — validation report

Date: 2026-08-29
Branch: `momcilo/kda_perf_model`
Implementation range: `ed6cab68943` through `0fef8b1211b`
Base: `ee7353a69eb38a2da9e059b276cf546c0dc46811`
Hardware: Blackhole, device 0, 110 available worker cores

## Verdict

PASS. The independent Python formulas passed all 23 cases, the integrated
`ttnn` build and branch-range checks passed, all 13 collected KDA production
performance pytest items passed on Blackhole, and one exact Tracy run for each
of the seven KDA APIs produced model fields that reconcile with the Python
model.

Per explicit user direction, validation is pytest-only. No C++ gtests were
added or run. The first host-only command used ambient `pytest` and stopped at
collection because that Python lacked `graphviz`; the corrected command below
uses the repository `python_env` and passed.

## Static, build, and formula validation

Commands:

```sh
pre-commit run --from-ref ee7353a69eb --to-ref HEAD
cmake --build build_kda_perf_model --target ttnn -j 16

env PYTHONPATH=/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model:/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/ttnn:/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/tools \
    LD_LIBRARY_PATH=/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/build_kda_perf_model/ttnn:/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/build_kda_perf_model/tt_metal:/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/build_kda_perf_model/tt_stl:/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/build_kda_perf_model/tt_metal/third_party/umd/lib \
    LOGURU_LEVEL=INFO TT_LOGGER_LEVEL=ERROR TT_METAL_LOGGER_LEVEL=ERROR PYTEST_ADDOPTS=--capture=sys \
    python_env/bin/pytest tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_kda_performance_model.py -vv
```

Results:

- branch-range pre-commit: PASS
- incremental `ttnn` build: PASS (`ninja: no work to do`)
- independent Python formula/model tests: 23 passed in 0.06 s
- coverage includes all seven work formulas, production and overflow values,
  `G=1`, all fidelities, non-square harvested grids, DRAM sum/dedup/alias and
  decimal rounding, L1, physical padding, invalid fallbacks, realtime clock
  conversion, and all three utilization ratios

## Integrated Blackhole performance pytest

Command:

```sh
env PYTHONPATH=/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model:/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/ttnn:/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/tools \
    LD_LIBRARY_PATH=/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/build_kda_perf_model/ttnn:/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/build_kda_perf_model/tt_metal:/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/build_kda_perf_model/tt_stl:/localdev/mvasilijevic/tt-metal.worktrees/kda_perf_model/build_kda_perf_model/tt_metal/third_party/umd/lib \
    LOGURU_LEVEL=INFO TT_LOGGER_LEVEL=ERROR TT_METAL_LOGGER_LEVEL=ERROR PYTEST_ADDOPTS=--capture=sys \
    scripts/run_safe_pytest.sh --run-all tests/ttnn/nightly/unit_tests/operations/experimental/kda -k production_performance -s -vv
```

Result: `13 passed, 252 deselected in 6.63s`; `SAFE_PYTEST_RESULT: PASS`.

| API / case | measured ns | FPU ns | DRAM/roofline ns | FPU / DRAM / roofline util. % | Accuracy evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| affine scan `sp1-tp8` | 97,231 | 1,232 | 26,112 | 1.27 / 26.86 / 26.86 | PCC 0.999958 |
| affine scan `sp2-tp4` | 74,879 | 1,056 | 27,648 | 1.41 / 36.92 / 36.92 | PCC 0.999987 |
| affine scan `sp4-tp2` | 64,712 | 704 | 30,720 | 1.09 / 47.47 / 47.47 | PCC 0.999998 |
| prepare recurrence | 25,405 | 32 | 736 | 0.13 / 2.90 / 2.90 | perf contract only |
| QKV Conv1D `single-block` | 86,561 | 99 | 1,554 | 0.11 / 1.80 / 1.80 | perf contract only |
| QKV Conv1D `multiple-blocks` | 46,894 | 198 | 3,108 | 0.42 / 6.63 / 6.63 | perf contract only |
| QKV Conv1D `asymmetric-split` | 51,963 | 58 | 907 | 0.11 / 1.75 / 1.75 | perf contract only |
| recurrent scan | 361,944 | 28,963 | 301,056 | 8.00 / 83.18 / 83.18 | perf contract only |
| reduce affine transforms | 45,942 | 2,049 | 18,432 | 4.46 / 40.12 / 40.12 | A 0.999989; B 0.999987 |
| sigmoid RMSNorm `sp1-tp8-local` | 154,631 | 7,452 | 92,176 | 4.82 / 59.61 / 59.61 | timing band passed |
| sigmoid RMSNorm `sp2-tp4-local` | 155,267 | 7,452 | 92,176 | 4.80 / 59.37 / 59.37 | timing band passed |
| sigmoid RMSNorm `sp4-tp2-local` | 155,234 | 7,452 | 92,176 | 4.80 / 59.38 / 59.38 | timing band passed |
| recurrence summary | 302,532 | 77,039 | 245,760 | 25.46 / 81.23 / 81.23 | perf contract only |

Every item reported `ideal_ns = max(ideal_fpu_ns, ideal_dram_ns)`. The
prepare and sigmoid cases also reported the expected omitted-SFPU result counts
of 25,344 and 7,925,760 respectively.

## Tracy reconciliation

Each exact node below was run with:

```sh
scripts/run_safe_pytest.sh --profile '<exact node>' -q
```

using the same `PYTHONPATH`, `LD_LIBRARY_PATH`, and logger environment from the
integrated command. All seven exact pytest items passed and each produced a new
CSV. `PM IDEAL`, `PM COMPUTE`, and `PM BANDWIDTH` match that API's realtime
Python roofline, FPU, and DRAM values exactly.

| API / exact case | Tracy report directory | PM ideal / compute / bandwidth ns | input / output BW slots |
| --- | --- | ---: | ---: |
| affine scan `sp1-tp8` | `2026_08_29_09_03_30` | 26,112 / 1,232 / 26,112 | 3 / 1 |
| prepare recurrence | `2026_08_29_09_03_50` | 736 / 32 / 736 | 5 / 7 |
| QKV Conv1D `single-block` | `2026_08_29_09_04_09` | 1,554 / 99 / 1,554 | 6 / 3 |
| recurrent scan | `2026_08_29_09_04_30` | 301,056 / 28,963 / 301,056 | 8 / 2 |
| reduce affine transforms | `2026_08_29_09_04_50` | 18,432 / 2,049 / 18,432 | 2 / 2 |
| sigmoid RMSNorm `sp1-tp8-local` | `2026_08_29_09_05_10` | 92,176 / 7,452 / 92,176 | 3 / 1 |
| recurrence summary | `2026_08_29_09_05_31` | 245,760 / 77,039 / 245,760 | 7 / 2 |

Reports are under `generated/profiler/reports/<directory>/`. The summary's two
output bandwidth entries are both `0.0`, as required for its height-sharded L1
outputs. Other slot counts and nonzero DRAM bandwidth values match declared
physical tensor traffic. Tracy FPU utilization also reconciles with its device
kernel duration; clock-derived compute time is independently covered by the
Python injected-clock tests rather than equated across profiler clock sources.

## Known non-blocking observation

The ambient `pytest` executable failed before collection with
`ModuleNotFoundError: No module named 'graphviz'`. This is an environment
selection error, not a model or test failure; the repository
`python_env/bin/pytest` command above passed all 23 tests.
