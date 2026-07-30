# Stage Review

Verdict: more-work-needed

## Required Work

- P2: The final manifest does not cover all evidence it accepts
  Evidence: `final_manifest.md` accepts
  `bounded_modulo_tail_cache_integrity.json`, the human prefill reports, and
  the modern/legacy profiler failure logs through its gate table and
  `profiler_summary.md`. None of those files is present in
  `final_manifest.sha256`. The manifest also omits `profiler_summary.md`
  itself. Most correctness JSONs (PCC, batch-2, boundary, trace) contain no
  embedded provenance, so the manifest hash list is the only immutable link
  available for them. `sha256sum -c` succeeds for the entries that are listed,
  but it cannot validate the omitted accepted evidence. In addition, the
  manifest calls 00:19–00:37 UTC the frozen run window while the four fused
  timing artifacts were written at 00:40:03 UTC and the checksum file at
  00:40:14 UTC.
  Why this matters: The original contract and the prior P1 remediation require
  final-source hashes to cover all accepted correctness, stress, watcher,
  timing, and profiler evidence. The current manifest cannot fail closed if an
  omitted stress artifact, profiler limitation log, or summary changes, and
  its stated freeze window does not describe the final artifact set.
  Required next step: Regenerate the manifest/checksum inventory after the
  artifacts are frozen. Include every file cited by an accepted gate or by
  `profiler_summary.md`, especially
  `bounded_modulo_tail_cache_integrity.json`, both `prefill_report.txt` files,
  the relevant `final*.log`/legacy logs, and `profiler_summary.md`; correct the
  run/freeze timestamps. Add the promised host-side manifest validator so
  accepted artifacts missing from the checksum inventory fail the test.

- P2: Fused-specific tests and final documentation still do not fully match the delivered contract
  Evidence: `tests/test_fused_decoder.py` verifies class selection, dense
  fusion, setup folding, source exclusions, and inherited prefill dispatch,
  but it does not execute `_moe_prefill_tile_group`, assert its sparse-matmul
  arguments/32-token split, validate `_router_weights`, or validate artifact
  provenance/manifest completeness. This is materially less than the
  refreshed `AUTODEBUG.md` experiment plan and leaves the manifest omission
  above undetected. Documentation is also internally contradictory:
  `README.md` says the final watcher passed 7/7 while the actual log and
  manifest show 9 passed; its opening description says the class overrides
  dense and decode-MoE GeGLU but omits the delivered prefill-MoE override.
  `work_log.md` retains the obsolete “Watcher: 7 passed” evidence row before
  later saying 9 passed.
  Why this matters: The user explicitly requires tests that cover overrides
  and provenance and exact final documentation. The current static tests
  prove dispatch to a lambda, not the implementation contract that previously
  regressed, and the final docs disagree about the accepted watcher suite.
  Required next step: Add host-only coverage that executes or structurally
  validates the prefill tile-group override, canonical 32-token splitting,
  router override, and fail-closed manifest inventory/provenance. Reconcile
  README/work-log wording and watcher counts with the final 9-test log.

## Other Concerns

- The final full-attention b1 traced-decode median win is only about 0.00013 ms
  (0.004%). The contract asks for a win and the retained 101-sample medians do
  satisfy that literal comparison, but the samples are sequential rather than
  paired and the margin is far below ordinary host jitter. Preserve this as
  residual risk rather than describing it as a robust performance separation.
- `AUTODEBUG.md` is correctly retained as the pre-fix diagnosis, but readers
  must use the later manifest/work log for resolution; it should not be
  mistaken for the final status.

## Hard-Check Gaps

- No test checks that every artifact referenced by an accepted manifest gate
  appears exactly once in `final_manifest.sha256`.
- Correctness JSONs other than the context/performance artifacts do not embed
  decoder/test/build provenance; complete external checksum coverage is
  therefore essential.
- The profiler summary gives command templates with placeholders rather than
  exact parameter-node commands. The raw logs support Blackhole execution and
  the modern join failure, but exact command provenance is not normalized into
  the manifest.

## Anomaly Ledger

- Observed anomaly: Accepted evidence is omitted from the checksum manifest.
  Evidence: `bounded_modulo_tail_cache_integrity.json`,
  `profiler_summary.md`, prefill human reports, and profiler limitation logs
  are absent from `final_manifest.sha256`.
  Affected path: Final correctness/stress and profiler evidence identity.
  Control or comparison: Every listed checksum currently verifies.
  Likely subsystem: Evidence lifecycle and manifest generation.
  Investigation performed: Compared all manifest gate citations and profiler
  summary citations against the checksum inventory and ran
  `sha256sum -c`.
  Resolution: more-work-needed

- Observed anomaly: Frozen-window claim predates final timing artifacts.
  Evidence: Manifest says 00:19–00:37 UTC; fused timing JSONs have 00:40:03
  mtimes and the checksum inventory has a 00:40:14 mtime.
  Affected path: Final timing provenance.
  Control or comparison: Timing JSONs embed the current decoder/test hashes
  and 7/101 sample counts, and their listed checksums verify.
  Likely subsystem: Final documentation timestamps.
  Investigation performed: Compared manifest prose, file mtimes, embedded
  hashes, and checksum verification.
  Resolution: more-work-needed

- Observed anomaly: Final watcher count is documented as both 7 and 9.
  Evidence: README and an early work-log evidence row say 7; watcher log,
  remediation section, and manifest say 9.
  Affected path: Exact stage documentation.
  Control or comparison: The final watcher log explicitly reports 9 passed and
  postdates the final source/test edits.
  Likely subsystem: Documentation refresh.
  Investigation performed: Read README, work log, manifest, timestamps, and
  watcher tail.
  Resolution: more-work-needed

- Observed anomaly: Modern Tracy join fails for retained-trace replay.
  Evidence: Retained `final_*.log` files show the join failure; raw
  `cpp_device_perf_report.csv` files contain Blackhole/110 metadata and two
  nonzero whole-trace device-duration rows per case.
  Affected path: Traced-decode per-op reporting.
  Control or comparison: Synchronized host replay samples agree with the
  device-duration regime; capture topology is separately labeled.
  Likely subsystem: Tracy host-op/device-marker joining for retained traces.
  Investigation performed: Inspected profiler summary, final logs, prefill
  reports, and all four device-profiler CSV headers/rows.
  Resolution: controlled

## Scope Inspected

- Goal/skill paths:
  `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/graph-fusing/SKILL.md`,
  `.agents/skills/tt-device-usage/SKILL.md`, and the supplied Stage 02
  contract.
- Artifact paths:
  `doc/fused_decoder/stage_review.md`, `README.md`, `work_log.md`,
  `AUTODEBUG.md`, `final_manifest.{md,sha256}`, `profiler_summary.md`, all
  final PCC/batch/boundary/context/trace/timing/watcher artifacts,
  `tracy/final_ops_{sliding,full}_b1`,
  `tracy/final_{sliding_b1,full_b1,sliding_b32,full_b32}`,
  `doc/context_contract.json`, functional timing controls, and git
  status/diff.
- Code paths:
  `tt/fused_decoder.py`, `tt/functional_decoder.py`,
  `tests/test_fused_decoder.py`, and `tests/test_functional_decoder.py`.
- Commands run:
  Read-only `sed`, `tail`, `find`, `grep`, `git status`, `git diff`,
  `sha256sum -c`, `stat`, and small read-only JSON/inventory scripts. No
  server, TT device, reservation, watcher, profiler, or hardware test was run.

## Residual Risk

- The implementation-level prior findings appear repaired: the prefill MoE
  override preserves canonical 32-token splitting and uses operand-specific
  fused GELU; final PCC/boundary/context/trace artifacts postdate the source;
  watcher evidence postdates source and tests; all six final median rows win;
  and Blackhole/110/nonzero profiler equivalents are present.
- Stage-owned files are enumerated and separable from unrelated `.agents`,
  `.skillexp-STAGE-RUNNING`, and GPT-OSS dirt, but the stage must not checkpoint
  until this rereview's evidence/documentation findings are fixed and a later
  independent review returns clean-pass.
