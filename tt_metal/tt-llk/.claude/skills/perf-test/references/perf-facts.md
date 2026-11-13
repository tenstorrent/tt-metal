# Perf facts

The single home for facts that every perf skill relies on. Other skills link
here instead of restating them; correct a fact here, once.

## Run types

| Run type | Measures | Threads not measured |
|---|---|---|
| `L1_TO_L1` | the functional run, end to end, `LOOP_FACTOR` times | none |
| `UNPACK_ISOLATE` | unpack | math stands in by clearing the source dvalids unpack posts; pack idles |
| `MATH_ISOLATE` | math | unpack stands in by posting the source dvalids math consumes; pack idles |
| `PACK_ISOLATE` | pack | unpack and math idle; pack runs without waiting for math |
| `SFPU_ISOLATE` | SFPU (Quasar only) | as the test defines it |

`ALL_PERF_RUN_TYPES` (`helpers/perf/core.py`) is the Wormhole/Blackhole set;
`PERF_RUN_TYPES_QUASAR` (`helpers/llk_params.py`) is the Quasar set. An isolate
is a valid stage measurement only when it runs the same loop body as
`L1_TO_L1` (see `perf-test`). The stand-in conventions per architecture are in
`wormhole-blackhole.md` and `quasar.md` next to this file.

## Build kinds

Each of these changes the measured code. Never mix them in one report or one
comparison.

- **Speed of light.** `--speed-of-light` turns runtime parameters into
  compile-time constants (`SPEED_OF_LIGHT` is defined). The nightly
  **LLK perf** run measures it; the PR and merge perf gates build both sides
  without it, and the gate refuses to compare the two kinds. Normal builds also load the runtime-parameter struct, so they move more
  under layout changes.
- **Asserts.** CI builds perf with `TT_LLK_DISABLE_ASSERTS=1`, which drops
  `-DENABLE_LLK_ASSERT`. A local perf build without it measures code with
  asserts. `run_test.sh --perf` and `perf_compare_commits.sh` set it by
  default.
- **Perf counters.** `--enable-perf-counters` compiles counter code and is a
  separate report kind: compare its `mean(...)` columns only with other
  counter reports.
- **Coverage.** `--coverage` instruments the kernels and invalidates perf.
- **`LLK_PROFILER`** is defined in every profiler build, functional or perf.
  It means "profiler instrumentation is compiled", not "this is a perf
  config", so it must never select parameter values.

## CSV and normalization

- Markers: `INIT` and `TILE_LOOP` are the kernel's `START_PERF_MEASURE`
  zones (`counters.h`: the profiler zone plus, in counter builds, the counter
  window);
  `KERNEL` is the zone `trisc.cpp` wraps around every profiler build.
- The raw `<module>.csv` holds per-marker means. The `<module>.post.csv`
  divides the `mean(...)` and `std(...)` columns of `TILE_LOOP` rows by
  `loop_factor * tile_cnt`: cycles per staged tile. `INIT` and `KERNEL` rows
  stay unnormalized. Analysis uses `.post.csv`.
- Columns come from the template and runtime dataclass fields that the test
  passes. The per-module column set is in
  `helpers/perf/test_schemas.py`; the hardware-free header gate
  (`tests/hardware_free/perf/test_perf_header_gate.py`) fails on drift.

## Accounting and comparability

- The schema version is `PERF_TEST_SCHEMAS[<module>]["version"]` in
  `helpers/perf/test_schemas.py`, not a CSV column. Bump it once per PR,
  relative to main, when the module's columns change. Rows from different
  versions are not comparable.
- A `tile_cnt` accounting change makes rows incomparable without a version
  bump. When `tile_cnt` counts both operand tiles of a two-operand kernel,
  per-tile `MATH_ISOLATE` is about half a result tile; that halving is not a
  speedup. Do not shrink `TILE_CNT` to restore an older number: it is also the
  unpack loop bound and the stand-in dvalid count.
- At 16-bit, `dest_acc=Yes` can double the Dest handshakes relative to
  `dest_acc=No`, so a `dest_acc` delta mixes a datapath change with twice the
  section-done traffic.
- Compare like with like: architecture, build kind, `loop_factor`, marker,
  schema version, and report kind.

## Where reports land

- Workers write `<module>.<worker>.csv` and `.post.csv` into
  `/tmp/tt-llk-build/temp_perf_data/` (`$RUNNER_TEMP/tt-llk-build/...` under
  GitHub Actions). The worker is `gw0`, `gw1`, … under `-n`, else `master`.
- At session finish they are combined into
  `tt_metal/tt-llk/perf_data/runs/<tag>/<module>/` and `perf_data/latest`
  points at that run. The producer phase writes no report.
- `PERF_RUN_TAG` names the run (default `local-<utc timestamp>-<pid>` from
  `run_test.sh`). `run_test.sh` refuses a tag whose directory exists: a reused
  tag mixes rows from two runs.
  `PERF_KEEP_RUNS` (default 10) bounds how many runs are kept; raise it for
  A/B or repeat series, or older runs are pruned mid-analysis.
- `perf_data/` is gitignored. A report is identified by test, architecture,
  commit, and exact command.

## Determinism and layout sensitivity

- On Wormhole, one build normally gives identical cycles run after run. Spread
  within one build points to sticky state or a race, not noise.
- Two builds whose measured code is identical can still differ by about ±10%,
  with configs sitting at two or three fixed levels. An unrelated change,
  often to the idle threads' code, moves configs between the levels. The
  diagnosis recipe is "Layout sensitivity" in `perf-regression-check`.

## CI

- Workflow **LLK perf** (`llk-perf.yaml`) runs `tests/run_llk_perf_wormhole.sh`
  and `tests/run_llk_perf_blackhole.sh`: producer `-n 10`, consumer
  `--dist loadgroup -n 15`, `-m "perf and not accuracy"`, `--timeout=60`,
  speed of light for the scheduled run, `TT_LLK_DISABLE_ASSERTS=1`.
- CI cuts the perf sweep into groups of equal estimated time. It prefers
  measured per-test durations from earlier runs (`LLK_SPLIT_DURATIONS`,
  summed by `perf/sum_split_durations.py`) and otherwise falls back to the
  per-module seconds in `tests/perf_split_costs.json` and
  `tests/perf_split_costs_sol.json`. A module missing from that table counts
  as 1 s. When a PR adds a perf module or changes its build count a lot,
  update its entries and say so in the PR.
- The perf gates and `perf_compare_commits.sh` share
  `perf/regression_compare.py`: a point regresses only when it is slower by
  both more than the threshold and more than 30 cycles. The threshold depends
  on the architecture and the gate:
  - Blackhole: 2% on every gate; also the `perf_compare_commits.sh` default.
  - Wormhole PR gate (`llk-perf-gate.yaml`): 13%.
  - Wormhole merge gate (`merge-gate.yaml`): 13%, and 25% for `perf_matmul`
    and `perf_math_matmul`.
  - Wormhole nightly gate (`llk-perf-gate.yaml`): 10%, and 25% for the two
    matmul modules.

  Pass `--threshold` to `perf_compare_commits.sh` to match the gate when
  reproducing a Wormhole verdict locally (`0.25` for the matmul modules).
