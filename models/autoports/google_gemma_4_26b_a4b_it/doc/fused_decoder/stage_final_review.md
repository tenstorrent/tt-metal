# Stage Review

Verdict: more-work-needed

## Required Work

- P2: The final evidence freeze and fail-closed validator remain inconsistent
  Evidence: `final_manifest.md` freezes the run/artifact window at
  `2026-07-30 00:19–00:45 UTC`, but the accepted `watcher.log` completed at
  `00:46:51` and `final_manifest.sha256` was written at `00:47:12`. The
  checksum inventory verifies, but
  `test_final_manifest_is_complete_and_fail_closed` only requires a
  hand-written subset of ten files; it does not derive every artifact cited by
  the accepted-gates table or `profiler_summary.md`. In particular,
  `profiler_summary.md` says each decode modern-join failure is retained in
  `final_*.log`, while there is no `tracy/final_sliding_b1.log` and therefore
  no checksum for that claimed case.
  Why this matters: The prior rereview explicitly required a corrected final
  freeze and a validator that fails closed when accepted evidence is omitted.
  A stale window and subset-only inventory test can still accept an incomplete
  evidence set, so provenance closure is not yet demonstrated.
  Required next step: Correct the frozen window to include the final watcher
  and checksum generation; either retain and hash the missing sliding-b1
  decode join log or narrow the profiler claim with exact per-case log paths;
  and make the validator derive/check every artifact reference accepted by
  `final_manifest.md` and `profiler_summary.md`, rather than checking a fixed
  subset.

- P2: Final documentation still reports the obsolete five-test topology result
  Evidence: The live host-only suite collects and passes seven tests, including
  the new canonical split/tile-group/router contract test and checksum
  validation. Nevertheless, `final_manifest.md` reports
  `fused topology/dispatch | 5 passed`, and `work_log.md` says
  `Final host fused tests: 5 passed`. README and watcher counts are now
  correctly reconciled at 9/9.
  Why this matters: These files are the final stage record and are treated as
  claims. Their accepted gate must describe the live suite exactly, especially
  because the two added tests are the remediation for the prior review.
  Required next step: Update the manifest and work log to report the final
  seven-test suite, then regenerate the checksum inventory/validation after
  the documentation is frozen.

## Other Concerns

- The full-attention batch-1 traced-decode median advantage remains only about
  0.00014 ms (roughly 0.004%) in sequential samples. It satisfies the literal
  six-of-six comparison but is not a robust performance separation.
- `profiler_summary.md` still uses placeholder profiler commands rather than
  exact parameter node IDs. The retained raw Blackhole/110-worker reports
  support the stated timing regime, so this is an evidence-quality concern
  rather than a new correctness blocker.

## Hard-Check Gaps

- The manifest validator does not parse the accepted-gate and profiler-summary
  references, so additions or omissions outside its hard-coded set do not fail.
- No single recorded command reconstructs the complete final checksum
  inventory from the accepted evidence declarations.

## Anomaly Ledger

- Observed anomaly: The accepted watcher and checksum postdate the declared
  frozen window.
  Evidence: `final_manifest.md` ends the window at 00:45 UTC;
  `watcher.log` has mtime 00:46:51 and `final_manifest.sha256` has mtime
  00:47:12.
  Affected path: Final evidence provenance and watcher-clean gate.
  Control or comparison: The watcher hash in README and the checksum inventory
  matches the live watcher log, and the watcher reports 9 passed.
  Likely subsystem: Evidence lifecycle/documentation.
  Investigation performed: Compared manifest prose, filesystem timestamps,
  live hashes, watcher tail, and checksum verification.
  Resolution: more-work-needed

- Observed anomaly: Final host topology tests are documented as five but run as
  seven.
  Evidence: Host-only pytest collected and passed 7 tests; manifest and work
  log retain `5 passed`.
  Affected path: Final topology/provenance gate reporting.
  Control or comparison: The live test source contains seven test functions
  and its SHA-256 matches the manifest header and checksum inventory.
  Likely subsystem: Documentation refresh.
  Investigation performed: Read the test source and ran the host-only suite.
  Resolution: more-work-needed

- Observed anomaly: Decode profiler prose claims every modern-join failure log
  is retained, but the sliding-b1 final log is absent.
  Evidence: `profiler_summary.md` refers to each `final_*.log`;
  `tracy/final_full_b1.log`, `final_sliding_b32.log`, and
  `final_full_b32.log` exist and are hashed, while
  `tracy/final_sliding_b1.log` does not exist.
  Affected path: Sliding-attention batch-1 decode profiler limitation evidence.
  Control or comparison: Its raw device-profiler CSV exists, is hashed, reports
  Blackhole/110 metadata, and has nonzero trace duration.
  Likely subsystem: Profiler evidence inventory.
  Investigation performed: Compared profiler-summary claims, directory
  contents, and checksum entries for all four decode cases.
  Resolution: more-work-needed

- Observed anomaly: The prior implementation and watcher findings appear
  repaired.
  Evidence: Live source overrides dense, router, decode-MoE, prefill chunk, and
  prefill tile-group paths; canonical 32-token splitting and three sparse
  matmuls are present; all manifest-listed hashes verify; watcher postdates
  source and tests and reports 9/9.
  Affected path: Stage-02 fused decoder correctness and final-source sanitizer
  coverage.
  Control or comparison: PCC, boundary, context, trace, tail-cache, functional
  baseline, fused timing, profiler, and watcher artifacts in the manifest.
  Likely subsystem: Graph fusion and evidence remediation.
  Investigation performed: Static source/test inspection, hash verification,
  artifact inspection, host-only pytest, and timestamp comparison.
  Resolution: fixed

## Scope Inspected

- Goal/skill paths:
  `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/graph-fusing/SKILL.md`,
  `.agents/skills/tt-device-usage/SKILL.md`, the supplied Stage 02 contract,
  and prior `stage_review.md` / `stage_rereview.md`.
- Artifact paths:
  `doc/fused_decoder/README.md`, `work_log.md`,
  `final_manifest.{md,sha256}`, `profiler_summary.md`, watcher output,
  correctness/context/trace/tail-cache JSONs, all functional/fused timing
  JSONs, prefill CSV/TXT reports, decode raw profiler CSVs, and retained
  join/legacy logs.
- Code paths:
  `tt/fused_decoder.py`, `tt/functional_decoder.py`,
  `tests/test_fused_decoder.py`, and `tests/test_functional_decoder.py`.
- Commands run:
  Read-only `sed`, `find`, `grep`, `git status`, `git diff`, `stat`,
  `sha256sum`, small read-only JSON inspection, and host-only
  `pytest -q .../tests/test_fused_decoder.py`. No server, TT device,
  reservation, watcher, profiler, or hardware test was run.

## Residual Risk

- The implementation-level prior findings are repaired and the final matrix
  reports fused wins in all six required rows, but the smallest decode win is
  within plausible host jitter.
- Stage-owned paths are identifiable and separable from unrelated `.agents`,
  `.skillexp-STAGE-RUNNING`, and GPT-OSS dirt. The stage should not checkpoint
  until the final evidence/documentation issues above are fixed and a later
  independent review returns `clean-pass`.
