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

Facts this skill relies on (build kinds, normalization, accounting, report
paths, determinism) are in
[perf-facts](../perf-test/references/perf-facts.md).

## Related skills

- `perf-test`: create, unify, or repair a perf test. Use it first when the
  test does not exist, hangs, reports implausible metrics, or drifts from
  functional.
- `perf-coverage`: check that the perf sweep matches functional.
- `run-test`: the runner this skill uses.
- `perf-regression-check`: compare against the branch point.
- `perf-parameter-impact`: analyze a validated `.post.csv`.

## 1. Establish the sweep

Read the perf test, `tests/python_tests/perf_[op].py` or
`tests/python_tests/quasar/perf_[op]_quasar.py`, and determine:

- which `run_types` the test reports;
- `loop_factor`, `tile_cnt`, and the other axes recorded as columns;
- how many variants `@parametrize` produces (`run-test` `count` with
  `--perf`);
- the architecture: the `quasar/` directory or a `*_[arch].py` suffix implies
  it, otherwise ask;
- the module's schema version in `helpers/perf/test_schemas.py`.

If the perf sweep may have drifted from functional, run `perf-coverage`
first. Decide scope before running: a narrowed sweep (`--k`) is right for
debugging; a report meant for analysis covers the full sweep. Never narrow the
sweep in the test file to make a run finish.

## 2. Run the sweep

Use `run-test` with `--perf`. It runs the producer and consumer phases the
way CI does (`-m perf`, `--mode perf` for SFPU sweep modules,
`TT_LLK_DISABLE_ASSERTS=1`, no `maxfail`, `--dist loadgroup -n 15` on
silicon), serializes device or simulator access, diagnoses hangs, and prints
the report directory:

```bash
bash .claude/scripts/run_test.sh run --worktree "$PWD" --arch <arch> \
  --test perf_[op].py --perf [--speed-of-light] [--run-tag <tag>]
# ... === RUN_LLK_TESTS_PERF_REPORT === <tt-llk>/perf_data/runs/<tag>
```

Choose the build kind deliberately:

- `--speed-of-light` matches the nightly **LLK perf** run; the PR and merge
  perf gates build without it. Match the CI numbers you compare against,
  and never mix speed-of-light and normal rows.
- `--perf-counters` (plus `--dump-perf-counters` for raw values) produces a
  counter report, a separate report kind. Run it as its own sweep. Perf
  counters work on Quasar too: four banks in every NEO and one l1_client
  event CSR in place of the L1 bank
  (`docs/performance_counters/hardware_reference.md`).
- `--keep-asserts` measures code with `LLK_ASSERT`; its numbers are not
  comparable with CI.
- Do not use coverage builds. Never hand-edit a CSV; fix the test or rerun.

`tests/run_llk_perf_wormhole.sh` and `tests/run_llk_perf_blackhole.sh` are
the exact CI invocations, for reference.

## 3. Locate the files

The consumer combines per-worker files into
`perf_data/runs/<tag>/<module>/<module>.csv`, `<module>.post.csv` and, for
counter runs, `<module>.counters.csv`; `perf_data/latest` points at the
newest run. After an aborted run, partial per-worker files are in
`/tmp/tt-llk-build/temp_perf_data/`. No report at all usually means the
consumer never reached session finish, or every selected test was skipped
(`RUN_LLK_TESTS_PERF_REPORT === none`). Details:
[perf-facts](../perf-test/references/perf-facts.md), "Where reports land".

## 4. Validate the artifact

1. **Schema.** A `PerfSchemaError` means one test emits different columns
   across its sweep (usually a parameter that is `None` for some values) or
   two ops share one module. Fix the test.
2. **Row count.** Reconcile rather than assert equality. Start from
   `selected variants × markers` (`INIT`, `TILE_LOOP`, `KERNEL`), then
   subtract skipped or deselected variants; duplicate keys are rejected,
   never merged. A counter run's `.counters.csv` has only `INIT` and
   `TILE_LOOP`. An unexplained shortfall means an aborted or partly skipped
   sweep.
3. **Duplicate keys.** `combine_perf_reports()` warns when it collapses rows
   sharing a (sweep, marker) key. Differing metrics on a collapsed key are
   either run-to-run variation or a parameter that changes the kernel
   without being recorded as a column. Resolve which before shipping.
4. **Plausibility** of `marker == TILE_LOOP`. This is the checklist every
   perf skill uses.
   - Failure signals: values near 2048, 4096 or 8192; an isolate orders of
     magnitude above the real stage; a healthy first variant followed by
     slow ones; a passing run that leaves the next run slow. These point to a
     handshake, stand-in count, section-done or wait-mask bug: switch to
     `perf-test`, "Repair a run type".
   - Across the run types of one config, when isolates are slices of
     `L1_TO_L1` (`perf-test`): every isolate is at or below `L1_TO_L1`, and
     `L1_TO_L1` is at or above the slowest isolate. The gap is the
     cross-thread handshake, largest for one-tile blocks. An isolate above
     `L1_TO_L1` does extra work: a prefill, a different loop shape, or a
     stand-in slower than the real partner.
   - An isolate whose stage does not exist in that config (`MATH_ISOLATE`
     with unpack-to-dest and math compiled out) measures an empty run: a
     value near the zone overhead. Exclude it and say so; it is not a stage
     cost.
   - Format and Dest-mode changes have bounded, explainable effects.
   - Determinism: rerun one build (`run-test --rerun --perf`). On Wormhole
     its cycles are normally identical; spread within one build points to
     sticky state or a race.
5. **Accounting.** Note the schema version and whether `tile_cnt` counts
   operand tiles ([perf-facts](../perf-test/references/perf-facts.md),
   "Accounting") before comparing with an older report.
6. **Freshness.** Compare CSV columns with the current test axes. Missing
   axes mean the report predates the test; regenerate instead of analyzing.
7. **Completeness, by report kind.** A timing report has a
   `mean(<run type>)` column for every requested run type and a
   `TEXT_SIZE(<run type>)` column for `L1_TO_L1`, `UNPACK_ISOLATE`,
   `MATH_ISOLATE` and `PACK_ISOLATE`. A counter report adds the
   `<RUN_TYPE>_..._pct` counter columns; compare its means only with other
   counter reports.

Never present metrics from a run whose pytest phase failed.

## 5. Record provenance

Report back, and keep alongside the CSV when it is archived:

- test file, module name, and schema version;
- architecture and `CHIP_ARCH`;
- repository commit;
- the exact `run_test.sh` command, including `--speed-of-light`, counter
  flags, `--keep-asserts`, and the run tag;
- run types and markers present;
- row count and output paths.

## Refresh and compare

- Each run lands in its own `perf_data/runs/<tag>/`, so a rerun cannot
  overwrite an earlier report or leave part of one behind. Name runs with
  `--run-tag`, and read the run you mean, not `latest`, when comparing two.
- For series of more than 10 runs, pass `--keep-runs` so older runs are not
  pruned mid-analysis.
- Compare like with like
  ([perf-facts](../perf-test/references/perf-facts.md), "Accounting"). For
  two commits, use `perf-regression-check`.

## Checklist

- [ ] Test, architecture, schema version and intended scope confirmed.
- [ ] Ran through `run-test --perf`; both phases completed.
- [ ] Build kind deliberate and uniform (speed of light, counters, asserts).
- [ ] The report read is the run just done (`RUN_LLK_TESTS_PERF_REPORT`).
- [ ] Single schema, row count reconciled, duplicate warnings reviewed.
- [ ] `TILE_LOOP` plausibility and determinism checked.
- [ ] Columns match the current test sweep.
- [ ] Provenance recorded.
- [ ] Report handed to `perf-parameter-impact` for analysis.

## Example triggers

- "Generate a perf report for `perf_matmul_quasar.py`."
- "Refresh the SFPU unary report and tell me what changed."
- "Why is there no `.post.csv` for this test?"
- "Is this report stale?"
