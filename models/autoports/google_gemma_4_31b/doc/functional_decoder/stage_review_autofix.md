# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The work log's final checkpoint section still refers to the earlier
  `stage_review_rereview.md` clean pass and commits predating the resumed
  autofix. This is not a pre-commit stage defect by itself: the live worktree is
  intentionally awaiting this review and the original goal requires the final
  stage-owned commit and SHA logging after a clean pass.

## Hard-Check Gaps

- The evidence manifest hashes the final runtime/test sources and filtered perf
  CSVs, but does not hash the exact-context, watcher, or human-readable report
  artifacts. Existing paths, timestamps, raw contents, and source hashes are
  sufficient to validate the correctness remediation and regenerated tables;
  expanding the manifest would be useful but is not required.

## Anomaly Ledger

- Observed anomaly: the first autofix rereview found CSV-export diagnostics in
  the four files labeled as human-readable performance reports.
  Evidence: the earlier state of each canonical
  `tracy/<kind>/<mode>/*_perf_report.txt` contained nine status lines and no
  operation table.
  Affected path: required performance evidence packaging.
  Control or comparison: all four files were regenerated from their existing
  canonical `*_ops.csv` inputs without `--csv`, using the matching start/end
  signposts plus `--no-color --no-summary --no-advice`. Sliding/full prefill
  reports are now 40 lines and decode reports 58 lines. Each contains the
  `ID`, `OP Code`, and `Device Time` table, followed by 25-device-op/0-host-op
  or 43-device-op/0-host-op totals of 3,521, 4,254, 2,577, and 2,911 microseconds.
  Likely subsystem: evidence rendering command.
  Investigation performed: inspected every regenerated text report, its
  filtering messages, table header/rows/footer, canonical CSV/signposts, and
  the work-log/autofix provenance update.
  Resolution: fixed.

- Observed anomaly: the resumed review's maximum-context decode was TTNN
  prefill-versus-TTNN decode of the same already-filled token.
  Evidence: `stage_review_resume.md` and `AUTODEBUG.md`.
  Affected path: maximum-context paged decode and mutable trace inputs.
  Control or comparison: the replacement tests prefill exactly 262143 periodic
  real-weight history tokens, change the stable captured token allocation to a
  distinct final token, replay at absolute position 262143, and compare directly
  to a reduced HF one-query oracle. The oracle first matches stock HF at
  0.99999997/0.99999999 PCC. Final replay PCC is 0.999406 sliding and 0.998875
  full, with wrong-position RMSE controls worse in both cases.
  Likely subsystem: old evidence semantics.
  Investigation performed: traced the new prefill extent, captured allocation,
  replay order, oracle math, page mapping, PCC/RMSE operands, and both raw logs.
  Resolution: fixed.

- Observed anomaly: padded lanes in bounded sliding prefill could wrap modulo
  1024 and overwrite live circular-cache slots.
  Evidence: `AUTODEBUG.md` and the old fill behavior described there.
  Affected path: non-aligned sliding prefill followed by decode.
  Control or comparison: `_fill_bounded_sliding_cache_exact` bulk-fills only
  complete valid tiles and performs device-side sequential paged updates for
  the valid tail. Real-weight 1025/1057 tests then perform a distinct HF-vs-TTNN
  decode and read the formerly clobbered K/V slot. Decode PCC is at least
  0.997702 and K/V PCC is at least 0.999885.
  Likely subsystem: padded paged-fill ownership under circular modulo addressing.
  Investigation performed: inspected valid/bulk/tail bounds, update-position
  tensors, page-table slicing, modulo arguments, cache readback mapping, and the
  focused/standard logs.
  Resolution: fixed.

- Observed anomaly: a captured trace could have baked stale token or current
  position contents.
  Evidence: the prior batch-32 test recopied identical values.
  Affected path: traced paged decode, RoPE lookup, and cache addressing.
  Control or comparison: new tests mutate the same stable token, uint32 RoPE
  position, and int32 cache-position allocations at seeded random positions and
  across 1023->1024 for both layer kinds. All changed replays exceed 0.998948
  PCC, correct RMSE is materially below stale-output RMSE, and repeated changed
  replay is bitwise deterministic. A separate `TT_METAL_WATCHER=10` run passes
  all four cases.
  Likely subsystem: old test coverage rather than retained runtime behavior.
  Investigation performed: inspected buffer construction/copies, capture/replay
  order, sequential HF cache reference, sensitivity controls, and watcher logs.
  Resolution: fixed.

- Observed anomaly: the watcher console contains words such as `exception` in
  compiler flags and runtime configuration.
  Evidence: `logs/autofix_watcher_trace_mutation.log`.
  Affected path: watcher evidence classification.
  Control or comparison: the 1987-line generated watcher log has normal
  attach/check/detach records and no fatal exception, assert, invalid NOC,
  CB/L1/stack overflow, sanitizer, or hardware-fault finding; pytest reports
  four passing changed-input trace cases.
  Likely subsystem: benign configuration/provenance text.
  Investigation performed: scanned both watcher console and generated watcher
  log for fatal and suspicious terms and inspected shutdown records.
  Resolution: controlled.

- Observed anomaly: Tracy reports a warning that its optional host `.tracy`
  copy was unavailable.
  Evidence: all four `logs/perf_*_autofix.log` files.
  Affected path: profiler host-trace convenience artifact.
  Control or comparison: every selected pytest node passes, final-source capture
  timestamps postdate the source/test edits, canonical ops CSVs contain matching
  start/end signposts, and filtered `Device Time` sums reproduce 3.521/4.254 ms
  prefill and 2.577/2.911 ms decode. The reports are on Blackhole with 110 worker
  cores.
  Likely subsystem: optional Tracy GUI trace-copy path, not device profiling.
  Investigation performed: inspected profiler logs, raw signposts, filtered CSV
  rows/totals, timestamps, and manifest hashes.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: `.exp_run/multigoal_logs/01-01-functional-decoder.prompt.txt`;
  `.agents/skills/functional-decoder/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`;
  `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/context_contract.json`; `doc/functional_decoder/README.md`;
  `work_log.md`; `evidence_manifest.json`; `AUTODEBUG.md`; `AUTOFIX.md`;
  `stage_review_resume.md`; final standard/focused/exact-context/watcher/profiler
  logs; generated watcher log; all four canonical Tracy raw, CSV, console, and
  text report groups.
- Code paths: `tt/functional_decoder.py` and
  `tests/test_functional_decoder.py`. Later-stage `fused_decoder` files were
  recognized as out of scope and not reviewed as stage-01 deliverables.
- Commands run: read-only `sed`, `rg`, `find`, `wc`, `tail`, `stat`, `sha256sum`,
  `awk`, `git status`, `git log`, and `tt-perf-report --help`; no TT device,
  pytest hardware, reset, server, profiler capture, or implementation/test
  mutation was performed.

## Residual Risk

- Runtime/test hashes exactly match `evidence_manifest.json`, and all four
  filtered performance CSV hashes match. The full correctness suite reports 25
  passed and eight explicitly gated long/performance tests; the gated exact
  context and performance nodes have separate passing logs.
- The exact full-attention HF oracle uses periodic history and a reduced
  one-query implementation rather than stock HF over 262143 history tokens.
  Its near-identity stock-HF validation at short length, official real weights,
  real absolute positions, all logical history positions, wrong-position
  sensitivity control, and above-threshold final PCC make this acceptable.
- Static source inspection and the signposted device-operation windows support
  the clean runtime-fallback claim. The fallback test includes the newly added
  exact-tail helper; no forbidden host conversion appears in the delivered
  runtime source.
- The human-readable tables were regenerated from already captured canonical
  ops CSVs, so this evidence-only repair appropriately required no additional
  hardware run. No required work remains in the inspected stage-01 scope.
