# Stage 04 resumed independent review

Verdict: `clean-pass`

## Required work

None.

## Other concerns

None. Later Stage 05 changes improve performance but do not reveal a latent
Stage 04 correctness or contract defect.

## Hard-check gaps

None.

## Anomaly ledger

### Ethernet watcher instrumentation

- Observed anomaly: full Ethernet watcher instrumentation could not run.
- Evidence: `evidence/watcher_full_failed.log` records a 27,792-byte active-
  Ethernet watcher image exceeding the 25,600-byte buffer; the no-inline retry
  passed tests but timed out restoring an instrumented router.
- Affected path: watcher instrumentation and teardown, not normal TP4
  execution.
- Control: worker/NoC watcher with only Ethernet instrumentation disabled
  passed both layer kinds. `evidence/watcher.xml` reports two passes and
  `evidence/watcher_device.log` has no watcher error signature.
- Likely subsystem: Blackhole active-Ethernet watcher firmware capacity and
  lifecycle.
- Investigation: full, no-inline, stale-device, partial-reset, coordinated-
  reset, and Ethernet-disabled runs are retained.
- Resolution: controlled.

### Profiler slow rows

- Observed anomaly: final profiler reports flag decode MLP rows and prefill
  attention projections as `SLOW`.
- Evidence: final `tt-perf-report` tables show decode BFP4 rows at roughly
  48-50% modeled utilization and prefill attention at roughly 20-25%.
- Affected path: decoder MLP and prefill attention performance.
- Control: the precision-locked 8/12/21/24-core decode sweep selected 24
  cores; the prefill 24-core program beat auto and other measured legal
  programs; final whole-layer TP4 latency beats the single-chip baseline in
  all four measured modes.
- Likely subsystem: matmul geometry and DRAM/compute utilization.
- Investigation: geometry, placement, QKV-core, fabric-link, attention-
  placement, and replicated-versus-fractured topology candidates were
  measured.
- Resolution: controlled.

### Matplotlib host warning

- Observed anomaly: `evidence/final_latency.log` reports a Matplotlib
  configuration-directory permission warning.
- Affected path: optional host plotting configuration only.
- Control: JUnit reports four passes; benchmark output records all samples and
  medians.
- Resolution: controlled.

## Scope inspected

- Original goal: `.exp_run/multigoal_logs/01-04-multichip-decoder.prompt.txt`.
- Skills: `$stage-review`, `$multichip`, and `$tt-device-usage`, plus section
  3.3 of `tech_reports/LLMs/llms.md`.
- Immutable checkpoint: `e1a3f724877`; implementation commit
  `683adda7a3d`.
- Code: the checkpoint versions of `tt/multichip_decoder.py` and
  `tests/test_multichip_decoder.py`.
- Evidence: context contract, run manifest, correctness, latency, geometry,
  fractured-boundary, watcher, and all four final Tracy report families.
- Commands: read-only `git show`, `git ls-tree`, `git diff`, `git status`,
  `rg`, `sed`, `wc`, and `sha256sum`. No hardware or write command was run by
  the reviewer.

The reviewer independently verified all manifest hashes. The Stage 04 final
suite has 11 passes and 15 opt-in skips, with separate passing evidence for the
exact-context, benchmark, geometry, and topology probes. The dirty live
worktree was explicitly recognized as a later Stage 05 optimization overlay,
not as part of this checkpoint review.

## Residual risk

- Stage 04 validates representative sliding and full decoder layers rather
  than a loaded 60-layer stack; full-model execution is intentionally outside
  this stage.
- Ethernet watcher coverage remains unavailable because of the documented
  instrumentation limit, while worker/NoC coverage is clean.
- The 262,144 context contract is supported by full-cache allocation,
  absolute-position trace, and conservative memory accounting; the reviewer
  did not rerun a populated TP4 262,144-token prefill.
