# Stage 3 optimize checklist assessment

Stop rule 2 fired: last two kept decode changes gained0.716% and0.263%.
The remaining untried performance items below are hand-off notes; no correctness
or acceptance gate is waived. See work_log.md for commands and candidate numbers.

| Checklist item | Decision and evidence |
|---|---|
| Real weights, warmed traced decode/prefill, context and logical lengths | Complete: baseline/final JSON, latency_comparison.csv, context_32768_0/3.json; logical127/128/129/257/4095/4096/4097;262144 unchanged |
| Initial topology audit | Complete: topology_audit.md and work-log operation table precede knob tuning |
| Best correct path and integrated final | Complete: candidate_ranking.json; final0/3 vs baseline; selected policy rerun within noise, no screening number substituted |
| Advice enabled and same-run accounting | Complete: tracy/layer_0/3 phase reports, perf_summary.json, perf_conclusions.md; remaining advice is explicit stop-rule handoff |
| Short/long/boundary prefill and physical kernel sizes | Complete: latency_comparison.csv and perf_conclusions.md; full-model TTFT not applicable |
| Dominant matmul precision-locked geometry | Measured: sparse_search/2, projection_search_retry and prefill_search; remaining per-role geometries untried at stop |
| DRAM sharding / Blackhole reader sweep | Kept: three readers/block4; readers1/2/3 and blocks8/16/32 tested,64 physical L1 failure after smaller legal variants |
| Packed QKV versus split | Kept packed: projection_search_retry and current split_projection_search reader/block controls |
| SDPA/KV contract / reduced KV dtype | Existing native paged SDPA preserved,32K smoke pass; reduced dtype untried at stop, no capacity change |
| Residual through norms/adds | Kept width sharding and explicit MoE output reshard: residual_search; removal measured slower |
| Logical batch separate from tile padding | Complete: battery0/3 B32, padded_trace_regression and page_extent_regression |
| Cumulative contract and integration | Complete: context_contract.json, hard_gates.json, full current-source battery |
| Row-parallel output decomposition | Untried at stop; measured output-projection geometry and residual layout retained |
| Persistent CCL | Not applicable: single device decoder, no CCL |
| Packed/split MLP gate-up | Kept packed: shared controls plus expert_split_search four tuned legal separate candidates |
| Phase-specific activation shards | Decode sharding and residual measured; further attention shard variants untried at stop |
| Precision crossed with geometry | Complete measured subset: BFP8/LoFi sparse blocks/cores/subblocks and BFP8/HiFi2 DRAM reader/block variants; BFP4 geometry candidate only |
| Canonical precision and attention separate from MLP | BFP8 default, expertsLoFi/shared-attentionHiFi2; sparse/projection searches, sensitive inherited BF16/FP32 exceptions documented |
| Actual target activations | natural_final0/3/34 and natural_stress34 use pinned HF-propagated real text activations |
| Policy reaches measured ops | hard_gates.json validates actual default and BFP4/LoFi profiler rows |
| MoE active-expert execution | Preserved exact8 indexed sparse decode; existing grouped prefill; no dense-all-expert fallback |
| LM head/sampling/full-model gap/qualitative suite | Not applicable: decoder-only stage; no generated text or full model |
| Evidence and safety | Trace allocation tracking includes program cache; Watcher separated from profiler; fuser and bounded serial jobs; zero resets; independent review required before completion |

Further untried items and expected gains (where known) are in work_log.md under
`Not tried (stop rule)`. They include recurrent prefill replacement, grouped MoE
repartitioning, selector/softmax fusion, reduced KV dtype, further physical-M
families and output decomposition. No unsupported claim of optimality is made.
