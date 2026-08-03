# Stage Review

Verdict: more-work-needed

Review cut: the live Stage-02 state containing the completed
`deepseek_moe_fast_reduce_nc_fused` assessment. While this report was being
prepared, the stage owner began changing the sequence-128 MoE implementation
and adding a permanent sequence-128 test. Those post-finding changes have no
updated correctness, watcher, latency, or profiler evidence and do not convert
this verdict; they require a fresh independent review after remediation is
complete.

## Required Work

- P1: The selected sequence-128 MoE prefill path is below the functional PCC
  bar, and the in-progress replacement is not validated.
  Evidence:
  `candidate_fused_reduce_prefill_tiled.json` records the four-32-token fused
  reduction at PCC 0.9876487399 and, critically, records the selected fallback
  through the same full public-layer active-expert reference at PCC
  0.9876476411. The artifact explicitly records `acceptance_pcc: 0.995`.
  Sequence 128 is the mandatory measured prefill workload. This is not an
  internal-only reduction metric: it is the public layer output checked through
  the functional `_assert_pcc(..., threshold=0.995)` path. The whole
  sequence-128 fused-reduction candidate is correctly rejected at PCC
  0.4088181766, but reverting to a result that is merely equivalent to the
  selected fallback does not establish that the fallback is correct.
  `pytest_results.xml` and `watcher_pytest_results.xml` each contain 19 passing
  cases, with MoE prefill PCC at sequence 1025 and sequence 33 but not sequence
  128. During this review `tests/test_fused_decoder.py` gained a sequence-128
  case, and `tt/fused_decoder.py` gained a tentative sequence-128 exact-sparse
  branch; both changes postdate the retained 19-case XMLs and the two MoE
  sequence-128 profiler windows. The README's statement that every PCC remains
  above 0.995 is therefore contradicted by the candidate artifact, and the
  current tentative replacement has no gate evidence.
  Why this matters:
  Stage 02 requires prefill PCC at the functional acceptance bar for every
  meaningful layer kind and requires the final code to reproduce the best
  correct measured path. A below-bar result on the exact performance workload
  is visible stage-critical correctness wrongness. It also invalidates closure
  based on the old 10.079501 ms selected prefill result once that implementation
  is replaced.
  Required next step:
  Finish one stable sequence-128 implementation and retain a permanent
  active-expert reference test that closes the observed sequence-128 failure at
  PCC >=0.995, including the previously sampled first/middle/last tokens. Rerun
  the complete normal and watcher suites on that exact source. Remeasure the
  affected sliding/full MoE sequence-128 wall and `tt-perf-report` windows and
  prove the corrected final default still beats the best correct functional and
  candidate paths. Rerun any non-aligned or capacity cases whose dispatch is
  changed, reconcile README/work-log/audit claims and artifacts, then obtain a
  fresh independent stage review.

## Other Concerns

- `doc/context_contract.json` remains valid for the unchanged 500000-token
  capacity contract, but its unqualified `independent_stage_review.verdict:
  clean-pass`, `pending_gates: []`, and `completion_status: complete` describe
  the earlier functional/context stage while the fused stage is not complete.
  The fused README supplies that scope, so this is not the correctness blocker,
  but the field remains easy to misread as the current Stage-02 verdict.
- The sequence-128 whole-shape and four-tile candidate artifacts are compact
  summary JSON rather than retained stdout/source snapshots. The recorded
  below-bar selected-fallback value is decisive enough for this verdict. The
  remediation should preserve the final test stdout and exact measured source
  so a fresh reviewer can re-derive closure without relying on prose.

## Hard-Check Gaps

- The retained normal and watcher evidence is internally sound for its 19
  cases: both XMLs report 19 tests, zero failures, zero errors, and zero skips;
  normal stdout preserves the PCC values; and the 20,734-line watcher log has
  no fatal/assert/illegal-NoC/timeout/hang/stuck/mailbox-error signature. It
  does not cover the newly added sequence-128 case or the implementation now
  being changed to address it.
- All nine original fused profiler windows are internally consistent: each raw
  CSV has one exact signpost pair, every in-window operation and device duration
  matches the filtered CSV, and every window has zero host operations. The two
  MoE prefill sequence-128 windows no longer prove performance for a replacement
  sequence-128 topology; the other seven windows remain consistent with their
  reviewed paths.
- The four-tile adaptation's 10.410248501 ms result is a valid performance
  rejection against the selected 10.079501 ms path, but its fallback-equivalent
  PCC is not a correctness pass. Performance rejection of that candidate does
  not waive correctness of the path selected instead.

## Anomaly Ledger

- Observed anomaly:
  The selected sequence-128 fallback has PCC 0.9876476411 against an acceptance
  threshold of 0.995.
  Evidence:
  `candidate_fused_reduce_prefill_tiled.json`;
  `candidate_fused_reduce_summary.json`; the absence of a sequence-128 MoE case
  from both retained 19-case XMLs; and the README's contradictory all-PCC-pass
  claim.
  Affected path:
  Sliding/full MoE prefill on the mandatory logical sequence 128 workload.
  Control or comparison:
  The four-tile fused adaptation is numerically equivalent at PCC 0.9876487399
  but slower at 10.410248501 ms; equivalence to a below-bar fallback is not a
  passing control. Existing sequence-1025 and sequence-33 final tests pass, but
  they do not control the observed sequence-128 result.
  Likely subsystem:
  Packed expert projection/reduction geometry and sequence-length dispatch.
  Investigation performed:
  Compared the new candidate artifacts with the public functional reference
  helper, current tests, both XMLs, current implementation dispatch, selected
  latency JSON, graph audit, README, work log, and prior reviews.
  Resolution:
  more-work-needed.

- Observed anomaly:
  A single whole-shape `deepseek_moe_fast_reduce_nc_fused` call is faster at
  sequence 128 but severely incorrect.
  Evidence:
  `candidate_fused_reduce_prefill_layer1_batch1.json` records 9.973846251 ms
  versus the selected 10.079501 ms, and
  `candidate_fused_reduce_prefill_seq128_pcc.json` records PCC 0.4088181766.
  Affected path:
  MoE prefill sequence 128.
  Control or comparison:
  Logical length 33 also fails at PCC 0.7588860077; a 1024-token call exceeds
  L1; four independent 32-token calls remove the catastrophic fused-reduction
  error but expose the selected fallback's separate below-bar PCC.
  Likely subsystem:
  Fused weighted-reduction token geometry/layout contract.
  Investigation performed:
  Recomputed the retained sample means, inspected the op binding/validation,
  checked current exact-32 dispatch and all relevant JSON/doc references.
  Resolution:
  The whole-shape candidate is correctly rejected; final sequence-128
  correctness remains more-work-needed as above.

- Observed anomaly:
  The exact 32-token fused weighted reduction is shape-sensitive but is retained
  at serving batch 32.
  Evidence:
  Normal stdout records PCC 0.9981931682; both MoE batch-32 raw/filtered
  profiler windows contain exactly one
  `DeepseekMoEFastReduceNCFusedDeviceOperation`; final wall results are
  8.272819/8.279496 ms and device totals are 8247.599/8233.539 us.
  Affected path:
  Sliding/full MoE traced decode at serving batch 32.
  Control or comparison:
  The pre-fusion packed all-expert results were 8.299395/8.292355 ms and the
  functional controls were 11.121584/11.129159 ms.
  Likely subsystem:
  One-tile weighted expert reduction.
  Investigation performed:
  Matched raw signpost windows to filtered rows and normal PCC stdout, and
  inspected the `token_count == 32` implementation gate.
  Resolution:
  controlled/retained for exactly 32 tokens.

- Observed anomaly:
  The original fused paged-cache update used overlapping K/V core grids.
  Evidence:
  `candidate_cache_update.json`, the current disjoint value-grid construction,
  dense batch-1/batch-32 PCC, profiler rows, and watcher output.
  Affected path:
  Decode cache updates for all representative layer kinds.
  Control or comparison:
  Current reviewed decode windows contain one required value reshard plus
  `PagedFusedUpdateCacheDeviceOperation`; cache-slot, determinism, and trace
  tests pass.
  Likely subsystem:
  Fused cache-update NoC/core-grid contract.
  Investigation performed:
  Compared implementation, tests, candidate summary, profilers, and watcher
  log.
  Resolution:
  fixed.

- Observed anomaly:
  Normal pytest emits a shutdown-only nanobind leak diagnostic, and the
  environment warns about firmware qualification, motherboard discovery, and
  low shared memory.
  Evidence:
  `pytest_full.log` and `final_tt_smi_list.log`.
  Affected path:
  Python binding shutdown and host/device environment.
  Control or comparison:
  All retained cases pass, devices close normally, four Blackhole p300c boards
  enumerate, and the watcher run detaches cleanly with no fault signature.
  Likely subsystem:
  Binding diagnostics/environment qualification.
  Investigation performed:
  Inspected complete normal-log shutdown, final inventory, watcher head/tail,
  XMLs, and signature scan.
  Resolution:
  controlled/non-blocking.

## Scope Inspected

- Goal/skill paths:
  `.agents/prompts/model_bringup_multigoal/01b-fused-decoder.txt`;
  `.agents/skills/stage-review/SKILL.md`;
  `.agents/skills/graph-fusing/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`.
- Branch/worktree:
  Live worktree on `skillexp-work-coherelabs_nor-p3`, base commit
  `dc2023c6bdc5b4740c39c4bdbefe6f271e1f4a39`; Stage-02 files are untracked.
  The implementation and test changed during review after the blocker was
  communicated, so this verdict intentionally requires a fresh post-fix cut.
- Artifact paths:
  README, work log, graph-fusing audit, `stage_review.md`,
  `stage_rereview.md`, all candidate JSON/text artifacts, all latency and
  capacity JSONs, normal stdout and both XMLs, watcher log and metadata, final
  device inventory, and all nine fused plus nine functional raw/filtered/table
  profiler controls under the fused/functional decoder documentation roots;
  `doc/context_contract.json`.
- Code paths:
  `tt/fused_decoder.py`, `tt/functional_decoder.py`,
  `tests/test_fused_decoder.py`, both fused wrappers and inherited functional
  test/perf/capacity harnesses; relevant fused-reduction, fused-gate,
  `moe_compute`, and `moe_gpt` bindings/validation and common MoE usage.
- Commands run:
  Read-only `sed`, `nl`, `rg`, `find`, `wc`, `stat`, `sha256sum`,
  `git status`, `git branch`, `git rev-parse`, and small read-only Python
  parsers for JSON/XML/CSV test counts, sample means, signposts, operation
  sequences, device totals, row matching, host-op counts, and candidate
  comparisons. No TT hardware command, test, benchmark, server, reset, trace
  capture, profiler capture, or implementation mutation was run.

## Residual Risk

- The rest of the delivered evidence is strong: exact serving batch-32 fusion
  is retained and correct; the whole sequence-128 candidate, sequence-33
  candidate, four-tile adaptation, and sequence-1024 L1 blocker are classified;
  cross-branch dense packing and exact-sparse batch 32 have measured
  rejections; advertised capacity is preserved; the original nine profiler
  windows and before/after gains are genuine; there are no measured host
  fallbacks; and the retained watcher run is clean.
- None of that controls the explicit below-bar sequence-128 selected-fallback
  PCC. Because the implementation is now being changed in response, correctness
  and performance closure must be established on one stable final source and
  reviewed afresh.
