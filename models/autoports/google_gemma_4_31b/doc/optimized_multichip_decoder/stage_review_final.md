# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The fused all-gather+matmul row-projection decomposition was rejected by a bound, not tried at Gemma's exact TP4 shapes under a compatible residual contract.
  Evidence: The stage records only an exact-shape fused matmul+reduce-scatter test (`tests/test_multichip_decoder.py:1101-1133`); there is no Gemma fused AG+matmul test or candidate artifact. The README and work log instead assert that AG+matmul can remove at most the measured gather launch and use the slower generic fractured-boundary result plus the MM+RS/Ring result to reject both fused endpoints (`README.md:59-69`, `work_log.md:149-154`). That is not the decomposition required by OPT-008: AG+matmul must gather the TP-local input, multiply by a repacked local-output weight, and carry the local hidden result through the compatible distributed norm/residual consumer before ranking the family. The inherited Stage 04 claim that the fused API requires eight devices is also not a current blocker. The live `AllGatherMatmulAsync` validator has rank/dim/program-config/shard-count constraints but no TP8 constraint (`ttnn/.../all_gather_matmul_async_device_operation.cpp:25-80`), and this checkout contains a same-hardware 1x4 Blackhole TP4 Ring probe that adapted its first output-subblock error and passed AG PCC 1.0 and matmul PCC 0.999993 (`models/autoports/qwen_qwen3_4b/doc/optimized_multichip_decoder/fused_all_gather_matmul_qwen_probe.log:94-99,184-186`).
  Why this matters: Both attention O and MLP down are material row-parallel boundaries in the final decode trace. The optimize contract explicitly says not to reject the gathered-input/local-output decomposition because MM+RS or a generic fractured-residual probe lost; fusion can change overlap and matmul geometry, so the claimed 20 us upper bound is not demonstrated. The user's completion contract also requires every applicable fused matmul-CCL family to be adapted/retried rather than deferred.
  Required next step: Add exact Gemma TP4 fused AG+matmul probes for the material O and down shapes. Repack/reshard each weight to local output width, adapt rank/layout/padding/program config after any first API error, and measure correctness plus traced whole-boundary/whole-layer latency. Measure the local-output result through the existing stack-compatible distributed norm/residual/next-projection contract without an immediate restore to replicated residual; separately account for a trailing gather only as the replicated-compatibility candidate. Try persistent/preallocated AG output state if the fused row is repeated and material. Keep the faster accepted coherent family or retain the current path with exact blocker/whole-family evidence, then rerun the final default gates and a fresh stage review.

## Other Concerns

- None. The prior reviews' four implementation findings are closed by current evidence: the declared 0.995 gate controls candidate selection, the packed-BFP8 MLP geometry sweep preserves the selected policy, CCL input/output dtype semantics are reconciled, and the redundant BFP8-to-BFP8 pre-CCL typecast is absent from the post-fix profiler.

## Hard-Check Gaps

- The optimized context test reaches absolute position 262,143 with the full cache allocation, determinism, replica equality, and finiteness, but does not repopulate the entire history. Prior-stage populated-context evidence remains linked and cache dtype/layout/capacity did not change, so this is not independently blocking this optimization review.
- The stage-owned commit and administrative status transition are explicitly pending until review and were not used to fail the verdict.

## Anomaly Ledger

- Observed anomaly: The second review found a same-dtype BFP8-to-BFP8 typecast before the MLP collective.
  Evidence: `stage_review_rereview.md`; pre-fix `tracy/final_selected` rows versus post-fix `tracy/final_noop_removed/{sliding,full}_decode/perf_report.csv`.
  Affected path: Final traced decode hot path before remediation.
  Control or comparison: Post-fix reports retain the necessary BF16-to-BFP8 attention cast and BFP8-to-BF16 boundary restores, but no BFP8-to-BFP8 `TypecastDeviceOperation` before MLP all-reduce.
  Likely subsystem: `_tp_allreduce` same-dtype conversion guard.
  Investigation performed: Source guard inspection, final CSV row inspection, checksum verification, and post-source-edit final latency/suite/context/watcher/profile artifact checks.
  Resolution: fixed.

- Observed anomaly: Current-topology BFP4 attention is faster but full-attention PCC is below the declared gate.
  Evidence: `candidates/review_cumulative_low_precision.log` records full PCC 0.992102801; `review_attention_bfp4_final.log` isolates full PCC 0.992305137, below 0.995.
  Affected path: Rejected precision candidate only.
  Control or comparison: QKV7 + BFP8 decode CCL passes sliding/full at 0.999802416/0.999718188.
  Likely subsystem: Full-attention projection precision sensitivity.
  Investigation performed: Cumulative and isolated current-topology real-weight reruns.
  Resolution: controlled.

- Observed anomaly: The initial Linear fused MM+RS probe hung and retained-scratch candidates hit L1 collisions.
  Evidence: `candidates/fused_mmrs_hang_triage.txt.gz`, `fused_mmrs_model_shape*.log`, `ring_l1_AUTODEBUG.md`, `ring_l1_AUTOFIX.md`, and historical failed XMLs.
  Affected path: Rejected candidates, not the final Linear persistent-all-reduce default.
  Control or comparison: Ring exact-shape MM+RS passed PCC 0.999963; adapted Ring and Linear families, final suite, watcher, and cleanup teardown pass.
  Likely subsystem: Fused Ring-kernel topology contract and persistent L1 placement.
  Investigation performed: Triage/reset recovery, Ring adaptation, shared tail-24 scratch, batch-32 lifecycle repair, and current-family reruns.
  Resolution: controlled/fixed for the selected default; it does not substitute for the untried AG+matmul decomposition above.

- Observed anomaly: The final Ring family gains only about 1-2 us decode while materially regressing sliding prefill.
  Evidence: `candidates/ring_final_selected.log` versus final Linear `evidence/final_latency.log`.
  Affected path: Rejected Ring topology family.
  Control or comparison: Same selected QKV7, MLP14, phase dtype, lifecycle, and layer kinds.
  Likely subsystem: Topology-dependent prefill communication cost and cross-run decode variation.
  Investigation performed: Complete current-policy Ring rerun after scratch reduction.
  Resolution: controlled for Ring versus Linear selection.

## Scope Inspected

- Goal/skill paths: supplied Stage 05 optimized-multichip-decoder contract; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/optimize/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; starting Stage 04 multichip README/work log.
- Artifact paths: Stage 05 README/work log/context contract/perf accounting/candidate matrix; both earlier reviews; raw review, geometry, topology, fused, precision, prefill-advice, lifecycle, and recovery logs/XML; post-fix final suite/latency/context/watcher evidence; `tracy/final_noop_removed` enriched CSV, advice tables, filtered CSVs, summaries, provenance, JUnit XML, and checksums.
- Code paths: live `tt/multichip_decoder.py`, `tests/test_multichip_decoder.py`, precision suffix change, fused MM+RS helper change, `AllGatherMatmulAsync` implementation/validator, and stage-owned diffs from starting checkpoint `e1a3f724877`.
- Commands run: read-only `git status/log/diff`, `find`, `rg`, `sed`, `stat`, SHA-256 verification, Python AST/JSON/XML/CSV parsing, profiler-row inspection, and independent device-time summation. No TT device, server, reset, or hardware test was run.

## Residual Risk

- Apart from the missing AG+matmul family, the final selected path has strong current evidence: 12 standard-suite passes, 4 final latency passes, 2 advertised-position passes, 4 separate watcher passes, and 4 final profiler windows. All post-fix hardware artifacts postdate the 12:04 source edit; profiler checksums verify; the device sums independently reproduce 428.14175/480.92775 us; and the final default reproduces PCC 0.999802416/0.999718188 with 0.464383/0.5175075 ms traced decode.
- Persistent-pool teardown is synchronized, identity-guarded, idempotent, and exercised before mesh close after trace release. No remaining lifecycle, fallback, non-aligned-length, context-capacity, watcher, or final-default provenance defect was demonstrated.
