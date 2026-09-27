# Candidate comparisons

All values are warmed traced **host-wall microseconds**, from actual4096/128, batch1, TP4. These are selection evidence; final headline numbers must come from final defaults. Full history, exact source hashes and PCC vectors are in candidate_summary.csv and each JSON/command/log.

| Family | Sliding decode us / minimum PCC | Full decode us / minimum PCC |
|---|---:|---:|
| baseline | 650.322 / 0.997611240 | 724.813 / 0.999447756 |
| Early QKV4 candidate + persistent L1 | 644.093 / 0.996739788 | 702.634 / 0.995366478 |
| Early QKV4 candidate CCL workers | 643.056 / 0.996739788 | 700.843 / 0.995366478 |
| BFP4 QKV split | 670.513 / 0.996739788 | 724.175 / 0.995366478 |
| carried mesh shard + fused QKV + BF8 MoE | 818.671 / 0.996668825 | 877.413 / 0.994962447 FAIL |
| carried mesh shard + fused MMRS | 816.101 / 0.996668825 | 887.982 / 0.994894863 FAIL |
| carried mesh shard + WO AGMM + persistent | 801.778 / 0.997143229 | 883.679 / 0.995501481 |
| L1 residual + L1 CCL | 660.910 / 0.996944984 | 723.955 / 0.995224571 |
| attention activation BF8, cumulative | 665.378 / 0.997088265 | 731.419 / 0.995616167 |
| shared activation BF8, cumulative | 653.255 / 0.996895437 | 712.759 / 0.995315199 |

The table above is historical candidate selection evidence, not the final
precision policy. Expanded real adjacent0->1 validation rejects sliding QKV4
and expertGate4 in combination. Real adjacent4->5 supports full QKV4 with WO8;
full WO4 also fails real batch32. The selected policy is sliding QKV8,
expertGate8, attentionCCL16; full QKV4, expertGate4, attentionCCL8. Both keep
expertDown4 and sharedGate4; sharedDown is BFP8 sliding and BFP4 full. Final-policy geometry and topology
remeasurements use the `final_policy_*` artifacts. The fused K44 regression
is repaired by respecting each22-tile gather slice; final-policy carried-shard
comparisons are the `repaired_sharded_*` artifacts below.

The sharded families consume localH704 through distributed norms/residuals. No immediate replicated-output restore is measured inside the layer. Harness gather is outside layer timing. Output-column WO AGMM uses localN704 and replaces output RS; QKV AGMM is a distinct fusion. Replicated residuals win the decode target even though sharded prefill can be faster.

All DRAM QKV/WO reader1/2/3 and wider four-core storage configurations lose whole-layer latency, including required weight zero-padding, input/output conversion and logical-width slicing. The actual multi-reader MeshDevice defect was fixed, built natively and regression-tested; see AUTOFIX_dram_mesh.md. BFP4 pad and N1 full grid errors were adapted and retried.

Shared prefill K17 (subblock1x2 and4x2, DRAM/L1 input) passes. Whole-layer differences are small (roughly0.1–0.4%) and do not improve the primary traced decode target; retain the existing prefill path. Gate/down sparse geometry and shared packed/split controls retain the prior correct final policy; new current-policy split shared/expert controls are included in this stage.

CCL worker1/2/4, buffer2, chunk1/4 controls were completed. Keep worker1 sliding and worker2 full; keep default buffers/chunk selection. Channel/chunk alternatives are not deferred. Adapted chunk API uses the input-owned mesh overload; original mesh overload lacks that keyword.

Numerical rejections: sliding combined QKV+WO BFP4 falls below0.995. Full combined QKV+WO BFP4 passes on replicated residuals but not the ordinary/MMRS sharded+BF8-MoE combination. Higher-precision WO restores the sharded controls. WO AGMM changes arithmetic decomposition and passes its own gate but is slower. The artificial0->5 stack (skipping four real layers) is kept only as diagnostic
evidence; its failures do not veto a policy. The real4->5 fixtures recompute
HF0..3 on the exact token history, and both4096/128 and33/128 clear0.995. See
AUTODEBUG_stack_precision.md and the adjacent-stack JSONs.

## Final-policy carried-shard fusion after the K22 repair

| Family | Sliding host decode µs / minimum PCC | Full host decode µs / minimum PCC |
| --- | ---: | ---: |
| qkv_agmm | 805.689 / 0.998281294 | 860.117 / 0.996207258 |
| wo_agmm | 809.286 / 0.998306491 | 883.768 / 0.996624403 |
| mmrs | 803.094 / 0.998281294 | 867.612 / 0.996111039 |

All preserve the localH704 residual through distributed norms and consumers;
no immediate replicated restore is timed. The final policies use sliding
QKV8/expertGate8 and full QKV4/expertGate4, with WO8 and MoE-BFP8 payloads
for these compatible sharded families. All lose to the selected default.
The original K44 corruption is preserved separately and is not a rejection.
