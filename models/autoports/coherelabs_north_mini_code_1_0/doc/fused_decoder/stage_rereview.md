# Stage Review

Verdict: more-work-needed

Review cut: the live Stage-02 state before
`candidate_fused_reduce_prefill_seq128_pcc.json` and its accompanying
documentation updates were added. After this verdict was communicated, the
stage owner reproduced the finding at PCC 0.4088181766 and added that evidence.
That remediation confirms the finding and is noted below, but it does not
retroactively turn this review into a clean pass; the changed state requires a
fresh independent re-review.

## Required Work

- P1: The reviewed snapshot did not correctness-classify the faster
  sequence-128 weighted-reduction candidate, so its prefill fusion audit was
  incomplete.
  Evidence:
  `candidate_fused_reduce_prefill_layer1_batch1.json` is explicitly linked from
  `candidate_fused_reduce_summary.json` as the
  `deepseek_moe_fast_reduce_nc_fused` prefill candidate. Its 20 warmed samples
  have mean 9.973846 ms, minimum 9.960733 ms, and maximum 10.012160 ms. The
  selected final fallback in `latency_prefill_layer1_batch1.json` has mean
  10.079501 ms and minimum 10.033567 ms, so every candidate sample is faster
  than every final sample and the mean advantage is 105.655 us (1.05%).
  At the review cut, however, the candidate artifact had no sequence-128 PCC,
  and `candidate_fused_reduce_summary.json` classified prefill only with an
  unrelated non-aligned length-33 PCC failure (0.758886) and a length-1024 L1
  overflow. The 1024-token allocation could not reject 128 tokens: the
  recorded 4,882,432 bytes/bank scales to about 610,304 bytes/bank at 128,
  below the recorded 1,461,504-byte availability, and the retained 128-token
  timing already proved that a candidate configuration ran. The TTNN op
  binding describes `[experts_k, 1, tokens, hidden]`, and its validation at
  `deepseek_moe_fast_reduce_nc_fused_device_operation.cpp:84-112` has no
  one-tile-only token restriction. Current code selects the fused op only for
  exactly 32 tokens (`tt/fused_decoder.py:338-369`). Sequence 128 therefore
  keeps the material transpose/reshape/route-multiply/reduce sequence: 2.035 +
  18.210 + 387.332 + 172.307 us in the sliding-MoE final profiler and 2.025 +
  18.060 + 385.013 + 181.317 us in the full-MoE final profiler.
  After the finding was reported, the stage owner added
  `candidate_fused_reduce_prefill_seq128_pcc.json`, which reproduces the exact
  measured-length candidate against selected tokens 0/63/127 and fails at PCC
  0.4088181766. The candidate code was reverted, and the summary/audit/README/
  work log now record that failure.
  Why this matters:
  Sequence 128 is the mandatory measured prefill workload. A dedicated fused
  op candidate that is reproducibly faster there cannot be rejected by PCC at
  a different shape or by an L1 limit at eight times the token count. This
  left graph-fusing Steps 4 and 5, the exhaustive-pattern gate, and the
  requirement that the final default reproduce the best correct candidate
  unresolved at the review cut.
  Required next step:
  The stage owner has now performed the missing test and retained the decisive
  failure plus reconciled documentation. Preserve that remediation and obtain a
  fresh independent re-review of the changed state. The new reviewer should
  verify that the recorded candidate actually dispatches the sequence-128
  fused path, that PCC 0.4088181766 comes from the accepted active-expert
  reference, that final code remains reverted to the correct fallback, and
  that no further adapted tile-aligned form is material and unassessed.

## Other Concerns

- `doc/context_contract.json` still contains an unqualified
  `independent_stage_review.verdict: clean-pass`, `pending_gates: []`, and
  `completion_status: complete`, while the fused README correctly says the
  Stage-02 independent re-review is pending. The capacity values themselves
  remain valid and unchanged; qualify that field as the functional/context
  review or update the stage-status wording when Stage 02 eventually passes.
- The dense cross-branch rejection is adequately supported for performance:
  all three linked artifacts contain 20 samples, and the candidate regresses
  prefill, decode batch 1, and decode batch 32. Its summary records candidate
  PCC only for prefill and decode batch 1, not batch 32. This is not a blocker
  because the batch-32 candidate is slower and therefore cannot be selected,
  but future candidate summaries should distinguish measured PCC regimes from
  timing-only regimes.

## Hard-Check Gaps

- The normal stdout log preserves exact PCC values and proves 19/19 cases
  passed. The watcher evidence is JUnit plus the 20,734-line watcher log rather
  than captured watcher pytest stdout, so it proves the same threshold
  assertions passed but does not preserve a second set of PCC decimals.
- At the review cut, the sequence-128 fused-reduction candidate had wall
  samples but no candidate profiler, exact candidate source/config snapshot,
  or PCC result. The post-review PCC artifact closes the central correctness
  gap with a decisive 0.4088181766 failure; a fresh reviewer must assess that
  changed evidence rather than this reviewer converting the historical verdict.
- The candidate summary JSONs for the exact-sparse, cache-update, plain-reduce,
  and split alternatives are inspectable and internally consistent, but only
  the cross-pack and fused-reduce timing artifacts retain raw sample arrays.
  This does not independently create more required work under the review
  skill's acceptance of summary JSON.

## Anomaly Ledger

- Observed anomaly:
  A retained sequence-128 fused-reduction run is uniformly faster than the
  selected final prefill path but is rejected using evidence from lengths 33
  and 1024.
  Evidence:
  `candidate_fused_reduce_prefill_layer1_batch1.json`,
  `candidate_fused_reduce_summary.json`,
  `latency_prefill_layer1_batch1.json`, `graph_fusing_audit.md:55,98-100`,
  and the final sliding/full prefill profiler rows.
  Affected path:
  Packed all-expert MoE prefill at the mandatory logical sequence 128.
  Control or comparison:
  Candidate mean 9.973846 ms versus final mean 10.079501 ms; at the review cut
  no candidate sequence-128 PCC existed. The post-review reproduction is
  0.4088181766.
  Likely subsystem:
  Weighted expert reduction shape/layout adaptation and graph-fusing candidate
  classification.
  Investigation performed:
  Recomputed every sample mean, compared complete sample ranges, inspected the
  final code branch and op validation, derived the 128-token L1 scale from the
  recorded 1024-token numbers, and re-derived the unfused reduction rows from
  both final profiler CSVs.
  Resolution:
  more-work-needed at the review cut; subsequently reproduced and remediated,
  pending a fresh independent re-review.

- Observed anomaly:
  The first independent review found that exact serving-batch weighted
  reduction and dense cross-branch packing were unassessed.
  Evidence:
  `stage_review.md`, `candidate_fused_reduce_summary.json`,
  `candidate_cross_pack_summary.json`, final code, and final batch-32 profiler
  CSVs.
  Affected path:
  MoE decode batch 32 and dense prefill/decode.
  Control or comparison:
  The exact-32 weighted reduction now passes PCC 0.998193 and appears as
  `DeepseekMoEFastReduceNCFusedDeviceOperation`; the 11264-wide dense
  cross-pack candidate regresses all three mandatory dense regimes.
  Likely subsystem:
  Dedicated-op discovery and shared-LHS graph rewrite.
  Investigation performed:
  Inspected current dispatch, candidate samples, normal stdout, and raw and
  filtered profiler rows.
  Resolution:
  fixed for serving batch 32; dense cross-pack rejection controlled. The
  distinct sequence-128 issue remains above.

- Observed anomaly:
  The original fused paged-cache attempt used overlapping K/V grids.
  Evidence:
  `work_log.md:19-22`, `candidate_cache_update.json`, and
  `tt/fused_decoder.py:72-93,198-244`.
  Affected path:
  Decode cache updates for every layer kind and batch.
  Control or comparison:
  Current code constructs a disjoint value grid and performs one required
  reshard. Dense batch-1 and batch-32 traced PCC pass, cache-slot and
  determinism checks pass, final profiler windows contain the fused update,
  and the watcher scan is clean.
  Likely subsystem:
  Fused cache-update NoC/core-grid contract.
  Investigation performed:
  Compared code, tests, normal stdout, final profiler rows, and watcher output.
  Resolution:
  fixed.

- Observed anomaly:
  Normal pytest reports a shutdown-only nanobind leak diagnostic.
  Evidence:
  `pytest_full.log:331-358`.
  Affected path:
  Python binding shutdown after the correctness suite.
  Control or comparison:
  All 19 cases pass, the log subsequently records normal device/cluster close,
  the watcher run passes 19/19, and watcher detach completes for all four
  devices.
  Likely subsystem:
  Nanobind reference-count diagnostics.
  Investigation performed:
  Inspected the complete log tail, JUnit files, and watcher tail.
  Resolution:
  controlled/non-blocking.

- Observed anomaly:
  `tt-perf-report` warns that specialized operations are unclassified.
  Evidence:
  The nine final `*_perf_report.txt` files.
  Affected path:
  Roofline categorization/advice.
  Control or comparison:
  Each raw CSV has exactly one matching signpost pair; all raw in-window global
  IDs and kernel durations match the filtered report; every in-window op is a
  device op; and all documented totals reproduce exactly.
  Likely subsystem:
  Profiler operation-category metadata.
  Investigation performed:
  Parsed all nine raw/filtered pairs and compared row IDs, durations, op
  counts, host-op counts, and totals.
  Resolution:
  controlled/non-blocking.

- Observed anomaly:
  The hardware logs warn about firmware 19.9.0 being newer than the latest
  fully tested 19.5.0 bundle, an unknown B850M-C motherboard, and low shared
  memory at test startup.
  Evidence:
  `pytest_full.log` and `final_tt_smi_list.log`.
  Affected path:
  Test environment.
  Control or comparison:
  Four Blackhole p300c boards enumerate, all normal and watcher tests pass,
  devices close/detach normally, and there is no watcher fatal/assert/illegal
  NoC/timeout/hang/stuck/mailbox-error signature.
  Likely subsystem:
  Environment qualification and host shared-memory provisioning.
  Investigation performed:
  Inspected test startup/shutdown, final inventory, JUnit, and watcher output.
  Resolution:
  controlled/non-blocking for this stage.

## Scope Inspected

- Goal/skill paths:
  `.agents/prompts/model_bringup_multigoal/01b-fused-decoder.txt`,
  `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/graph-fusing/SKILL.md`, and
  `.agents/skills/tt-device-usage/SKILL.md`.
- Branch/worktree:
  Live worktree on `skillexp-work-coherelabs_nor-p3`, base commit
  `dc2023c6bdc5b4740c39c4bdbefe6f271e1f4a39`; Stage-02 files are currently
  untracked, as are unrelated cluster-descriptor files.
- Artifact paths:
  Every file under
  `models/autoports/coherelabs_north_mini_code_1_0/doc/fused_decoder/`,
  including README/work log/audit, prior review, all correctness/candidate/
  latency/capacity JSON and XML/text artifacts, nine raw/filtered/table
  profiler sets, watcher generated logs, and final device inventory;
  plus a post-verdict spot-check of
  `candidate_fused_reduce_prefill_seq128_pcc.json` and its updated summary/
  audit/README/work-log references;
  `doc/context_contract.json` and the corresponding functional baseline
  latency/profiler artifacts used only as controls.
- Code paths:
  `tt/fused_decoder.py`, `tests/test_fused_decoder.py`,
  `tests/fused_decoder_perf.py`, and `tests/fused_decoder_capacity.py`;
  inherited functional decoder/test/harness methods used by those wrappers;
  the DeepSeek fused-reduction binding and validation; and relevant fused MoE
  gate contracts needed to assess the audit.
- Commands run:
  Read-only `sed`, `nl`, `rg`, `find`, `wc`, `stat`, `git status`,
  `git branch`, `git rev-parse`, `git diff --no-index`, JSON formatting, and
  small read-only Python parsers for JSON/XML/CSV means, cases, signposts,
  global IDs, kernel durations, host-op counts, and before/after totals. No TT
  hardware command, test, benchmark, server, reset, trace capture, profiler
  capture, or implementation mutation was run.

## Residual Risk

- Apart from the unresolved sequence-128 classification at the review cut, the
  remediation was strong:
  fused tests genuinely substitute `FusedDecoder`; normal and watcher suites
  pass 19/19 with no skip; direct dense batch-32 PCC is 0.9998535; the exact-32
  weighted reduction has PCC 0.998193 and is visible in both MoE batch-32
  profilers; non-aligned, cache, determinism, dynamic-history, representative
  layer, and real-weight checks pass; all advertised-capacity probes are
  finite; all nine final wall/device rows beat their controls; and final
  profiler artifacts are internally consistent with zero host ops.
- This reviewer did not run hardware, by design. The stage owner subsequently
  produced the missing on-device PCC and kept the correct fallback. That
  post-review change removes the plausible faster-correct-candidate risk if
  its provenance checks out, but only a separate fresh review can issue the
  required clean-pass verdict for the remediated state.
