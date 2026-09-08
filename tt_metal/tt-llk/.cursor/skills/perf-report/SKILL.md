---
name: perf-report
description: Generate, refresh, and validate an LLK performance report by running a perf sweep end to end and checking the resulting perf_data CSV. Use when asked to produce or refresh a perf report for an op, run a perf sweep, or when a report is missing, partial, stale, or implausible.
---

# Perf Report

## Goal

Produce a trustworthy performance report for one LLK perf test and hand it to
analysis with enough provenance to reproduce it.

A report is a build artifact, not a repository file: `perf_data/` is
gitignored. What identifies a report is the test, the architecture, the
commit, and the exact command that produced it. Record all four.

## Related skills

- `quasar-perf-test` — create or repair the perf test and its `PerfRunType`
  paths, and choose tile/dimension coverage. Use it first when the test does
  not exist, hangs, reports implausible metrics, or needs sweep-axis changes.
- `perf-parameter-impact` — analyze a finished `.post.csv`.
- `run-test` — the repository test-runner workflow.

## 1. Establish the sweep

Read the perf test — `tests/python_tests/perf_[op].py` or
`tests/python_tests/quasar/perf_[op]_quasar.py` — and determine:

- which `run_types` the test actually reports;
- `loop_factor`, `tile_cnt`, and the other axes recorded as columns;
- how many variants `@parametrize` produces;
- the architecture: the `quasar/` directory or a `*_[arch].py` suffix implies
  it, otherwise ask.

For Quasar, `compare_test_and_perf.py --dir quasar` is the sweep-audit against
the functional counterpart (composite `list` = matrix, `tuple` = tile shape).
Flag stale-report risk when current test axes are absent from the CSV.

Decide scope before running. A narrowed sweep (`-k`, `--op`) is right for
debugging; a report meant for analysis must cover the full intended sweep.
Never narrow the sweep in the test file to make a run finish.

## 2. Run the sweep

Use the two-phase producer/consumer flow, never a single serial invocation:

```bash
cd tests
CHIP_ARCH=<arch> pytest --compile-producer -n 10 -m perf ./python_tests/perf_[op].py
CHIP_ARCH=<arch> pytest --compile-consumer -n 15 -m perf ./python_tests/perf_[op].py
```

Prefer the `run-test` workflow where it applies; it serializes simulator
access and diagnoses hangs. `tests/run_llk_perf_wormhole.sh` and
`tests/run_llk_perf_blackhole.sh` show the exact CI invocation.

Rules:

- `--speed-of-light` turns runtime parameters into compile-time constants and
  changes measured cycles. CI passes it. Match CI when the report will be
  compared with CI numbers, and never mix speed-of-light and normal rows in
  one report.
- `--enable-perf-counters` produces a different, mutually exclusive kind of
  report. It compiles with `-DPERF_COUNTERS_COMPILED` (the WC build), which
  reduces `ZONE_SCOPED` to metadata only: the run emits no wall-clock
  `mean(<run type>)` columns, only `<RUN_TYPE>_..._pct` efficiency columns.
  It also writes to the same `<module>.csv` path, overwriting the timing
  report. Move the timing report out of the way first (see Refresh and
  compare), run counters as a separate sweep, and validate the result as a
  counter report. `--dump-perf-counters`
  additionally writes raw counter values to `<module>.counters.csv`.
- Perf counters are unavailable on Quasar. The build gate keeps the define off
  there and `counters.h` `#error`s if it ever slips through, so the flag
  yields no counter columns.
- SFPU sweep modules need `--mode perf`; the selector defaults to `accuracy`
  and deselects the perf sweep.
- Do not pass `--coverage`. Instrumentation invalidates perf numbers.
- Avoid `-x` on a report run. It aborts mid-sweep and the combined CSV is
  silently partial. Use it only while debugging.
- Never hand-edit a CSV. Fix the test or rerun.

## 3. Know where the files come from

- Each worker writes `<module>.<worker>.csv` and `<module>.<worker>.post.csv`
  into `/tmp/tt-llk-build/temp_perf_data/` when the module-scoped
  `perf_report` fixture tears down. Under GitHub Actions the root is
  `$RUNNER_TEMP/tt-llk-build` instead. The worker is `gw0`, `gw1`, … under
  `-n`, otherwise `master`. Look there for partial artifacts after an
  aborted run.
- `pytest_sessionfinish` calls `combine_perf_reports()`, which merges the
  per-worker files into `perf_data/runs/<tag>/<module>/<module>.csv`,
  `<module>.post.csv`, and `<module>.counters.csv`, sorts them, and deletes
  the per-worker files. Each run writes its own `runs/<tag>/` directory and
  `perf_data/latest` is repointed at it, so a rerun neither overwrites an
  earlier report nor leaves part of one behind. `PERF_KEEP_RUNS` (default 10)
  bounds how many runs are retained. `PERF_RUN_TAG` sets the tag; off CI it
  defaults to `local-<utc timestamp>`.
- The producer phase writes no report and skips combining.
- The raw CSV holds per-marker means. The `.post.csv` divides the `mean(...)`
  and `std(...)` columns of `TILE_LOOP` rows by `loop_factor * tile_cnt`,
  giving cycles per tile; `INIT` and `KERNEL` rows are left unnormalized.
  Analysis uses `.post.csv`.

No report at all usually means the consumer phase never reached session
finish, or every selected test was skipped.

## 4. Validate the artifact

1. **Schema.** A `PerfSchemaError` means one test emits different columns
   across its sweep — usually a parameter that is `None` for some values — or
   two ops share one module. Fix the test; do not work around it.
2. **Row count.** Reconcile rather than assert equality. Start from
   `selected variants × markers`, then subtract skipped or deselected
   variants; duplicate keys are rejected, never merged. Markers are the
   zones the kernel declares — `INIT` and `TILE_LOOP` in the perf sources,
   plus the `KERNEL` zone that `trisc.cpp` wraps around every profiler build.
   A counter report has no profiler-derived rows, so expect only the counter
   zones `INIT` and `TILE_LOOP` there. An unexplained shortfall means an
   aborted or partly skipped sweep; a shortfall you can attribute to skips or
   collapse is fine.
3. **Duplicate keys.** `combine_perf_reports()` warns when it collapses rows
   sharing a (sweep, marker) key. Differing metrics on a collapsed key are
   either run-to-run noise or a parameter that changes the kernel without
   being recorded as a column. Resolve which before shipping the report.
4. **Plausibility.** Inspect `marker == TILE_LOOP`. Each `L1_CONGESTION` stage
   should sit near its isolate. Values near 2048, 4096, or 8192, an isolate
   orders of magnitude above the real stage, or a healthy first variant
   followed by slow ones all indicate handshake or wait-mask bugs — switch to
   `quasar-perf-test`.
5. **Freshness.** Compare CSV columns with the current test axes. Missing axes
   mean the report predates the test; regenerate instead of analyzing.
6. **Completeness, by report kind.** A timing report carries a
   `mean(<run type>)` column for every requested run type, and a
   `TEXT_SIZE(<run type>)` column only for `L1_TO_L1`, `UNPACK_ISOLATE`,
   `MATH_ISOLATE`, and `PACK_ISOLATE`. `L1_CONGESTION` is deliberately absent
   from the code-size map, so a missing `TEXT_SIZE(L1_CONGESTION)` is correct
   rather than a defect. A counter report has no wall-clock means at all:
   check its `<RUN_TYPE>_..._pct` columns, expect only the `INIT` and
   `TILE_LOOP` markers, and note that its `.post.csv` is identical to the raw
   file because normalization only rescales columns named `mean(...)` and
   `std(...)`.

Never present metrics from a run whose pytest phase failed.

## 5. Record provenance

Report back, and keep alongside the CSV when it is archived:

- test file and module name;
- architecture and `CHIP_ARCH`;
- repository commit;
- the exact producer and consumer commands, including worker counts,
  `--speed-of-light`, and counter flags;
- run types and markers present;
- row count and output paths.

## Refresh and compare

- Nothing needs moving aside. Each run lands in its own `perf_data/runs/<tag>/`
  and the previous run is untouched, so a rerun that skips everything or dies
  before the consumer phase cannot leave an earlier CSV looking like the new
  result. Read the run you mean, not `latest`, when comparing two runs.
- A counter run is still a separate report kind from a timing run, and the two
  share no metric columns — but they now land in different run directories, so
  one no longer replaces the other.
- Compare like with like: same architecture, same speed-of-light setting, same
  `loop_factor` and marker, and the same report kind. Timing and counter
  reports measure different things and share no metric columns.
- Repeat a run before attributing a small delta to a code change.

## Checklist

- [ ] Test, architecture, and intended scope confirmed.
- [ ] Producer and consumer phases both completed without aborting.
- [ ] Coverage off; speed-of-light setting deliberate and uniform.
- [ ] The report read is the run you just did — `perf_data/latest`, or the
      `runs/<tag>/` you intended.
- [ ] `perf_data/runs/<tag>/<module>/` holds the raw and `.post.csv` files, plus
      counters when requested.
- [ ] Single schema, row count reconciled, duplicate warnings reviewed.
- [ ] Column expectations applied for the report kind actually produced.
- [ ] `TILE_LOOP` metrics inspected for plausibility.
- [ ] Columns match the current test sweep.
- [ ] Provenance recorded.
- [ ] Report handed to `perf-parameter-impact` for analysis.

## Example triggers

- “Generate a perf report for `perf_matmul_quasar.py`.”
- “Refresh the SFPU unary report and tell me what changed.”
- “Why is there no `.post.csv` for this test?”
- “Is this report stale?”
