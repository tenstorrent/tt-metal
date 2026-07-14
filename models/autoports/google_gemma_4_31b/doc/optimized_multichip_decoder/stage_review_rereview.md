# Stage Review

Verdict: more-work-needed

## Required Work

- P2: The selected decode path performs an avoidable BFP8-to-BFP8 typecast before the MLP all-reduce.
  Evidence: `_tp_allreduce` casts every partial whenever `communication_dtype != ttnn.bfloat16`, without first checking whether `partial.dtype` already equals `communication_dtype` (`tt/multichip_decoder.py:188-195`). The selected packed gate/up and down path produces a BFP8 MLP partial (`:596-603`), and decode selects BFP8 communication (`:387-400`, `:638-660`). The current selected profiler confirms the consequence: `sliding_decode/perf_report.csv` row ID 1649 and `full_decode/perf_report.csv` row ID 2763 are `TypecastDeviceOperation`, `BFP8 => BFP8`, on 14 cores immediately before the second `AllReduceAsyncDeviceOperation`, costing 1.744 us and 1.747 us respectively. This is distinct from the necessary attention BF16-to-BFP8 cast and the two necessary BFP8-to-BF16 boundary restores.
  Why this matters: Stage 05 is an on-device optimization stage whose contract requires avoidable decode movement/conversions to be removed and the final default to be reproduced after the selected graph is final. This same-dtype copy is in every traced token, has no documented ownership, trace, layout, or op-contract justification, and its measured cost is comparable to the sub-2-us topology differences used in the final Ring-versus-Linear discussion.
  Required next step: Skip the pre-CCL typecast when `partial.dtype == communication_dtype`, preserving the existing deallocation/ownership contract. Rerun the current no-override Linear final correctness, latency, watcher, and selected profiler evidence; verify the MLP CCL remains `BFP8, BFP8 => BFP8`, the redundant row is absent, trace replay and terminal pool cleanup remain safe, and update device/e2e accounting and headline numbers.

## Other Concerns

- None beyond the required hot-path conversion. The original three blocking findings are substantively remediated:
  - The declared `PCC_THRESHOLD = 0.995` now governs selection. Cumulative QKV7 + attention-BFP4 + BFP8 CCL failed full attention at 0.992103, attention-BFP4 alone failed full attention at 0.992305, and QKV7 + BFP8 CCL passed at 0.999802/0.999718. The promoted defaults are QKV block 7, BFP8 decode CCL, and BF16 prefill CCL; the final no-override run reproduces 0.999802/0.999718 PCC and 0.464928/0.519778 ms decode.
  - The corrected MLP sweep inherits the selected packed gate/up, BFP8-output policy and measures both MLP-only and whole-layer traces over 7/8/12/14/21/24/28/42/56/84 cores. Fourteen cores wins at 0.2166945/0.464358 ms; 24 cores measures 0.221979/0.467863 ms. The initial 7-core L1 failure was adapted from block 24 to legal block 12 and passes at 0.216744/0.468147 ms.
  - TTNN source and selected profiler rows agree that async `dtype` controls output while the fabric input pages use the actual input dtype. Both selected decode reductions are `BFP8, BFP8 => BFP8`; prefill remains BF16, and the public layer boundary is restored to BF16.

## Hard-Check Gaps

- The live implementation mtime (11:59 UTC) is later than final suite, profiler, and Ring artifacts. No behavioral contradiction was found: the live default policy, CCL rows, geometry, cleanup API, and Linear/Ring selections match the artifacts, including the redundant BFP8-to-BFP8 row. Treat this as a provenance weakness rather than a second blocker; the required rerun above will supersede it.
- The advertised-position test exercises absolute position 262,143, trace replay, finiteness, determinism, and replica equality without repopulating the entire history. Prior-stage populated-context evidence remains linked, cache layout/capacity is unchanged, and current non-aligned/paged transition tests pass, so this is not independently blocking this decoder optimization rereview.
- The pending local checkpoint commit and status finalization are the explicitly permitted administrative pending items and did not affect the verdict.

## Anomaly Ledger

- Observed anomaly: Current-topology attention BFP4 is faster but fails full-attention accuracy.
  Evidence: `review_cumulative_low_precision.log` records full PCC 0.992102801; `review_attention_bfp4_final.log` isolates full PCC 0.992305137, both below 0.995.
  Affected path: Rejected precision candidate only.
  Control or comparison: QKV7 + BFP8 CCL passes both layer kinds at 0.999802416/0.999718188.
  Likely subsystem: Full-attention projection precision sensitivity.
  Investigation performed: Cumulative and isolated current-topology real-weight reruns.
  Resolution: controlled.

- Observed anomaly: Packed 7-core MLP geometry initially exceeded L1.
  Evidence: `review_packed_mlp_geometry.log` retains the failure; `review_packed_mlp_geometry_7_adapted.log` passes the block-12 adaptation at PCC 0.999802416.
  Affected path: Rejected 7-core candidate only.
  Control or comparison: Adapted 7-core layer latency 0.468147 ms versus selected 14-core 0.464358 ms.
  Likely subsystem: Packed 2N gate/up circular-buffer capacity.
  Investigation performed: Reduced legal block width and reran MLP-only plus full-layer trace.
  Resolution: controlled.

- Observed anomaly: Native BFP8 MLP input semantics were previously described as BF16 CCL payload.
  Evidence: TTNN `all_reduce_async` source derives output spec from the requested dtype but configures the input CB from `input_tensor.dtype()`; final selected rows show both CCLs as `BFP8, BFP8 => BFP8`.
  Affected path: Selected decode CCL.
  Control or comparison: BF16 prefill rows remain BF16; final decode PCC passes.
  Likely subsystem: Documentation/API-semantics interpretation.
  Investigation performed: Source reconciliation, explicit phase policy, current profiler capture.
  Resolution: fixed for dtype semantics; more-work-needed only for the separate redundant same-dtype copy identified above.

- Observed anomaly: The final cumulative Ring14 family gains only 1.17/1.71 us decode while materially regressing sliding prefill.
  Evidence: `ring_final_selected.log` passes PCC 0.999802/0.999718 and measures 2.618772/2.160720 ms prefill and 0.463758/0.518072 ms decode, versus Linear 2.4067785/2.174885 and 0.464928/0.519778 ms.
  Affected path: Rejected Ring topology family.
  Control or comparison: Same selected QKV7, MLP14, phase dtype, lifecycle, and real layer kinds.
  Likely subsystem: Topology-dependent prefill communication cost and cross-run decode noise.
  Investigation performed: Complete current-policy Ring rerun after scratch reduction.
  Resolution: controlled; Linear retention is evidence-backed.

- Observed anomaly: Initial Linear fused MM+RS hung and early retained-scratch candidates hit L1 collisions.
  Evidence: `fused_mmrs_hang_triage.txt.gz`, `fused_mmrs_model_shape*.log`, `ring_l1_AUTODEBUG.md`, `ring_l1_AUTOFIX.md`, and historical failed XMLs.
  Affected path: Rejected candidates, not the selected Linear async-all-reduce path.
  Control or comparison: Ring exact-shape fused probe passed PCC 0.999963; adapted current families, final suite, watcher, and cleanup teardown pass.
  Likely subsystem: Fused Ring-kernel topology contract and persistent L1 placement.
  Investigation performed: Triage/recovery, Ring adaptation, tail-24 shared scratch, batch-32 lifecycle repair, and current-family reruns.
  Resolution: controlled/fixed for the selected default.

## Scope Inspected

- Goal/skill paths: supplied Stage 05 optimized-multichip-decoder contract; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/optimize/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; `tech_reports/LLMs/llms.md` section 4.
- Artifact paths: stage `README.md`, `work_log.md`, `stage_review.md`, `candidates/summary.csv`, raw `review_*` and `ring_final_selected.*` logs/XML, `evidence/final_{suite,latency}.*`, exact-context and watcher evidence, `perf_accounting.json`, and `tracy/final_selected` provenance/source/filtered CSV/advice reports/hashes.
- Code paths: live `tt/multichip_decoder.py`, `tests/test_multichip_decoder.py`, `doc/context_contract.json`, relevant TTNN minimal `all_reduce_async` source, and stage-owned diffs from starting HEAD `e1a3f724877`.
- Commands run: read-only `git status/diff`, `find`, `rg`, `sed`, `stat`, XML/CSV parsing, profiler sum checks, and SHA-256 verification. No TT hardware, server, implementation test, reset, or mutation command was run.

## Residual Risk

- The selected path otherwise has strong current evidence: 12 final-suite passes including cleanup, both layer kinds, non-aligned lengths, batch 32, mutable/repeated trace, replica/cache/layout checks and fallback audit; 4 final latency passes; 2 advertised-position passes; 4 separate watcher passes; and 4 selected profiler windows with verified hashes.
- Persistent-pool teardown is now synchronized, identity-guarded, released-state guarded, idempotent, and ordered before mesh close after trace release. No remaining lifecycle defect is demonstrated by the inspected evidence.
- A later rereview should use only artifacts collected after the same-dtype typecast removal and confirm updated accounting rather than inheriting the present headline numbers.
