# Stage Review

Verdict: clean-pass

## Required Work

None.

## Other Concerns

- The README's opening latency table preserves the earlier accepted run, while
  the `Current-pass refresh (2026-07-30)` section records the authoritative
  rerun. The fresh batch-1 prefill point is 2.082 ms versus 2.030 ms functional
  (a 2.6% regression), rather than the earlier 4.8% improvement. This does not
  violate the stage's primary batch-1 decode target or batch-32 non-regression:
  fresh traced decode improves from 1.053 to 0.522 ms at batch 1 and from 1.216
  to 0.723 ms at batch 32, while batch-32 prefill improves from 26.238 to
  20.241 ms. The refresh section states this discrepancy directly and does not
  use the older prefill point to justify a selected decode configuration.
- `git diff --check 58af598354e^ 58af598354e` reports CRLF-style trailing
  whitespace in generated compact CSV reports. This is generated evidence, not
  a source or runtime defect; the committed Python and markdown artifacts
  compile/read cleanly.

## Hard-Check Gaps

- The advisor was captured once at batch 32 rather than separately at logical
  batches 1 and 32. Both public decode cases present a one-tile (32-row)
  activation to the dense matmuls, and the report records activation rows 32.
  The exact BFP4/LoFi advisor seed and final default were then remeasured with
  real layer-0 weights and 200 traced replays at both logical batches:
  555.616/756.719 us for the 8-core seed versus 522.067/723.259 us for the
  16-core default. This controls the single-capture risk and satisfies the
  batch-specific candidate-decision requirement.
- The default suite's batch-32 tests use synthetic weights unless
  `PHI35_REAL_WEIGHTS=1`; however, the current-pass default and advisor-seed A/B
  logs explicitly exercise real weights at both batches, and the direct
  real-weight layer test remains in the default suite. Phi-3.5 Mini has one
  homogeneous dense decoder-layer kind.

## Anomaly Ledger

- Observed anomaly: Tracy reports device-profiler buffers full after the
  retained signposted decode window.
  Evidence: `tracy/final/current_pass_profile_b1_console.log` and
  `current_pass_profile_b32_decode_console.log`; the compact reports retain one
  complete 62-op batch-1 and 59-op batch-32 window.
  Affected path: profiler collection after the measured replay, not the
  unprofiled traced default.
  Control or comparison: independent 200-replay default logs reproduce 522.067
  us at batch 1 and 723.259 us at batch 32; both runs pass real-weight PCC.
  Likely subsystem: finite Tracy device-profiler marker capacity.
  Investigation performed: inspected the console ordering, compact
  signpost-filtered reports, op counts, and independent timing logs.
  Resolution: controlled.

- Observed anomaly: TTNN substitutes a round-robin width-sharded matmul output
  grid for the requested rectangular 16-core residual grid.
  Evidence: current correctness/A-B logs emit the substitution warning; the
  decode reports show the explicit boundary reshard and no host operations.
  Affected path: DRAM-sharded decode matmul output to rectangular sharded
  residual/norm boundaries.
  Control or comparison: carrying the computed grid into RMSNorm was previously
  adapted and rejected by its 22-core bounding-box validation; the explicit
  rectangular restoration passes real-weight PCC and is included in all
  whole-layer latency measurements.
  Likely subsystem: DRAM-sharded matmul factory output-grid identity.
  Investigation performed: compared runtime warnings, source memory configs,
  profiler movement rows, and the documented adapted candidate.
  Resolution: controlled.

- Observed anomaly: watcher console contains nanobind leak diagnostics during
  Python teardown.
  Evidence: `watcher_current_pass_console.log` records five tests passing,
  successful device close, and no watcher kernel assert/error/hang; generated
  `watcher.log` is clean.
  Affected path: Python binding teardown after optimized correctness runs.
  Control or comparison: ordinary correctness and separate profiler runs close
  the device successfully.
  Likely subsystem: nanobind teardown/reference accounting.
  Investigation performed: inspected ordering relative to pytest completion,
  watcher output, and device shutdown.
  Resolution: controlled.

- Observed anomaly: fresh batch-1 prefill is 2.6% slower than the fresh
  functional point although an earlier run improved it.
  Evidence: `logs/current_pass_functional_perf.log` reports 2.030315 ms and
  `logs/current_pass_default_b1.log` reports 2.082184 ms.
  Affected path: batch-1, sequence-128 warmed prefill only.
  Control or comparison: the README preserves both measurements, batch-32
  prefill improves 22.9%, and the primary traced decode target improves 50.4%.
  Likely subsystem: small-run timing variation and prefill-specific path cost,
  not default decode wiring.
  Investigation performed: compared fresh same-checkout functional/default
  logs and separate prefill profiler output rather than relying on the earlier
  headline table.
  Resolution: controlled; accurately disclosed and not a failed contract gate.

## Scope Inspected

- Goal/skill paths:
  - User's optimized-decoder contract supplied to the reviewer.
  - `.agents/skills/stage-review/SKILL.md`
  - `.agents/skills/optimize/SKILL.md`
  - `.agents/skills/shard-advise/SKILL.md`
  - `.agents/skills/shard-advise/SETUP.md` Part B
  - `.agents/skills/tt-device-usage/SKILL.md`
- Artifact paths:
  - `doc/optimized_decoder/README.md`
  - `doc/optimized_decoder/work_log.md`
  - `doc/optimized_decoder/shard_advise/report.json`
  - `doc/optimized_decoder/shard_advise/final_ir.mlir`
  - `doc/optimized_decoder/logs/current_pass_*`
  - `doc/optimized_decoder/tracy/final/*_current_pass.{txt,csv}`
  - `doc/optimized_decoder/watcher_current_pass*`
  - `doc/context_contract.json`
- Code paths:
  - `tt/optimized_decoder.py`
  - `tests/test_optimized_decoder.py`
  - Functional decoder performance harness used as the before control.
- Commands run:
  - `git status --short --branch`
  - `git show --stat --oneline --decorate 58af598354e`
  - `git diff --check 58af598354e^ 58af598354e`
  - `python -m py_compile .../tt/optimized_decoder.py`
  - Read-only `sed`, `grep`, `find`, `git log`, `git diff`, and JSON inspection
    over the implementation and evidence.

## Residual Risk

- The review did not open TT devices or rerun hardware, as required by the
  independent review contract. It therefore relies on committed current-pass
  logs for runtime evidence.
- The exact advisor program seed is slower than the retained default at both
  logical batches; the DRAM-sharded family is retained, while compiler-selected
  broad output grids are not. Future TTNN matmul-grid changes could alter that
  tradeoff and should trigger a fresh same-policy A/B.
- Long-context stress is explicit opt-in rather than part of the default test
  invocation, but committed current-pass logs cover 32769, 131071, 131072
  prefill and logical-context-131072 decode, and the context contract is not
  reduced.
