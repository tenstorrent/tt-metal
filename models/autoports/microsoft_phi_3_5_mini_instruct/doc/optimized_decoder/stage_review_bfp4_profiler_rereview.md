# Stage Review

Verdict: more-work-needed

## Required Work

- P2: Reconcile final profiler conclusions in README and work log.
  Evidence: The recollected final CSVs correctly show block-16 decode down at
  47.879/47.992 us with 51.2–51.3% modeled DRAM utilization and explicit
  block-8 prefill down at 136.433 us (32 cores, B1) and 768.119 us (64 cores,
  B32), with zero host ops. README still says the final decode down row is
  51–52 us at 47.7–47.8% utilization, and its checklist describes only the
  automatic prefill configs. The work log does not record the final profiler
  recollection or its stage-owned CSV provenance.
  Why this matters: The implementation and profiler evidence now satisfy the
  runtime gate, but the user explicitly requires README/work-log
  tt-perf-report conclusions. The current top-level conclusion contradicts the
  final artifacts.
  Required next step: Update README with 47.879/47.992 us,
  51.2–51.3% utilization, and the explicit adaptive prefill-down selection;
  record the final B1/B32 decode/prefill profiler CSVs and conclusions in the
  work log. No hardware rerun is needed.

## Other Concerns

- None.

## Hard-Check Gaps

- After this documentation-only correction and clean rereview, create the
  isolated local stage commit and record branch/SHA; never push.

## Anomaly Ledger

- Observed anomaly: Final profiler evidence and top-level documentation report
  different decode row timings/utilization and prefill selection.
  Evidence: Final CSVs and updated profiler summaries versus README/work log.
  Affected path: Final optimization report.
  Control or comparison: CSV rows match the current source configs exactly and
  report zero host ops.
  Likely subsystem: Documentation lag after profiler recollection.
  Investigation performed: Direct CSV/source/report comparison.
  Resolution: more-work-needed.

## Scope Inspected

- Final block-16 decode B1/B32 CSVs and summary
- Final explicit-prefill-down B1/B32 CSVs and summary
- Current optimized decoder source, README, work log, correctness/performance,
  watcher, precision, and geometry artifacts
- Read-only shell inspection only; no hardware or implementation edits

## Residual Risk

- All implementation, correctness, watcher, precision, geometry, performance,
  context, and final profiler gates are otherwise closed.
