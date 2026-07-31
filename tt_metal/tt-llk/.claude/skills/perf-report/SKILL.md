---
name: perf-report
description: Generate, refresh, and validate an LLK performance report by running a perf sweep end to end and checking the resulting perf_data CSV. Use when asked to produce or refresh a perf report for an op, run a perf sweep, or when a report is missing, partial, stale, or implausible.
user_invocable: true
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
  paths. Use it first when the test does not exist, hangs, or reports
  implausible metrics.
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
- `--enable-perf-counters` adds efficiency-metric columns;
  `--dump-csv-counters` additionally writes `<module>.counters.csv`.
- SFPU sweep modules need `--mode perf`; the selector defaults to `accuracy`
  and deselects the perf sweep.
- Do not pass `--coverage`. Instrumentation invalidates perf numbers.
- Avoid `-x` on a report run. It aborts mid-sweep and the combined CSV is
  silently partial. Use it only while debugging.
- Never hand-edit a CSV. Fix the test or rerun.

## 3. Know where the files come from

- Each worker writes `<module>.<worker>.csv` and `<module>.<worker>.post.csv`
  into `$ARTEFACTS/temp_perf_data/` when the module-scoped `perf_report`
  fixture tears down — `gw0`, `gw1`, … under `-n`, otherwise `master`.
- `pytest_sessionfinish` calls `combine_perf_reports()`, which merges the
  per-worker files into `perf_data/<module>/<module>.csv`,
  `<module>.post.csv`, and `<module>.counters.csv`, sorts them, and deletes
  the per-worker files. Combined files are overwritten in place.
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
2. **Row count.** Expect `variants × markers` rows, with markers `INIT`,
   `KERNEL`, and `TILE_LOOP`. A short file means an aborted or partly skipped
   sweep.
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
6. **Completeness.** Confirm a `mean(...)` and `TEXT_SIZE(...)` column exists
   for every requested run type.

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

- Copy an existing report elsewhere before rerunning; combined files are
  overwritten in place.
- Compare like with like: same architecture, same speed-of-light setting, same
  `loop_factor` and marker.
- Repeat a run before attributing a small delta to a code change.

## Checklist

- [ ] Test, architecture, and intended scope confirmed.
- [ ] Producer and consumer phases both completed without aborting.
- [ ] Coverage off; speed-of-light setting deliberate and uniform.
- [ ] `perf_data/<module>/` holds the raw and `.post.csv` files, plus counters
      when requested.
- [ ] Single schema, expected row count, duplicate warnings reviewed.
- [ ] `TILE_LOOP` metrics inspected for plausibility.
- [ ] Columns match the current test sweep.
- [ ] Provenance recorded.
- [ ] Report handed to `perf-parameter-impact` for analysis.

## Example triggers

- “Generate a perf report for `perf_matmul_quasar.py`.”
- “Refresh the SFPU unary report and tell me what changed.”
- “Why is there no `.post.csv` for this test?”
- “Is this report stale?”
