# Optimization coverage

The initial topology audit precedes the experiments in `work_log.md`. This table maps the completed search to compact numerical artifacts. Candidate timings are whole-layer **host** elapsed unless a device profile is explicitly named; the final headline uses complete-layer device windows. This is a dense text decoder, so routed-expert/`nnz` requirements do not apply.

| Family or advice | Evidence and action |
|---|---|
| Repeated same-input projections | Keep packed sliding QKV / full QK with tied raw V. Stage03 packed-versus-separate attention controls and stage04 correctly dtype-matched packed gate/up controls are retained. Split gate/up wins the agent batch workload; see `../multichip_decoder/mesh_plan.md` and its candidate table. |
| Row-parallel collectives and carried residual | Stage04 compares full reductions, RS/AG, distributed norm and sharded residual carry. Stage05 `output_ag_sharded_layer0_batch1.json` carries the reduced residual through its consumers; the final harness gather is outside timing. It loses at933.16us versus427.21us selected sliding decode. |
| Fused CCL + matmul | Stage04 fused MM-RS and input-AG-MM families; stage05 output-sharded AG-MM repacks real weights to local1536 channels, fixes fused-input dtype, and retries smallerK/N. `output_ag_layer0_batch1.json` passes at670.27us; adapted variant908.15us. Both lose to the selected DRAM-sharded path. |
| Persistent buffers / memory placement | `ccl_l1`, `ccl_pool`, `row_l1_slice`, `pooled_slice`, then `norm8pool` controls. Final two shared L1 AG buffers avoid conversions and per-layer L1 growth. Shared-workspace alias/replay checks establish lifetime. |
| Lower movement at O/down | Direct DRAM slice saves sliding but regresses full decode480.66→492.52/493.14us. L1 slice followed directly by DRAM restores full477.42us; pooled L1 AG further reduces473.67us before adding norm optimization. |
| Activation / CCL dtype | Independent `attn8` and `mlp8` controls for both kinds. `norm8poolccl8` crosses BFP8 CCL with the selected buffer/layout family;427.70/476.97us loses to BF16 selected427.21/475.04us. Prefill already uses BFP8 CCL/intermediates. |
| Weight precision and fidelity | `attn`, `mlp`, `down` × `hifi2`/`lofi` controls use BFP8 real weights against BFP4/LoFi baseline. All pass, all slower; selected BFP4/LoFi preserved. KV remains validated BFP8. Norms remain HiFi4/FP32. |
| DRAM-sharded decode geometry | All five roles use DRAM-sharded weights/matmuls. `q12/q24`, `gate12/gate24`, `o16`, `down22` change legal sender grids and K blocks under selected BFP4/LoFi. Full QK24 also crosses best layout (`norm8poolq24`); loses478.77 vs475.04us. No final block≤2. Reader microprofiles and prior whole-layer reader controls supplement this sweep. |
| Prefill matmul program configuration | `tt-perf-report` does not parse `MinimalMatmulConfig` and advises a missing program config despite runtime M4/K8/N16/subblock1×8/grid11×10. Stage04 K14, smaller-grid/K14/N8 and padded6144 controls are measured; K21 has a concrete L1 allocation failure after adaptation. Keep faster correct explicit config. |
| Norm activation sharding | Seven-column norm tried with corrected shape API, legal FP32 subblock and in-place output; still exceeded L1. AutoDebug/AutoFix derived eight-column width672/subblock3 and occupied-row planner. Both kinds and batch/tail boundaries pass. |
| Host gaps / prefill tracing | Supplemental `check_multichip_prefill_trace` measures capture cost and repeated traced prefill on real4096-token inputs; outcome recorded with final profile analysis. Decoder calls remain composable for caller-owned traces. |
| Async scheduling / topology | Final Ring async RS/AG,2 links,2workers/link,1channel buffer, persistent RS staging. Stage04 topology/scheduling controls and stage05 buffer/dtype controls supply measured comparison; `ccl_report.md` gives bytes and consumer contracts. |

`candidate_timings.md` provides per-candidate PCC and latency, and manifests preserve exact argv/source hashes. `short_tail_comparison.json` compares matching short/awkward-prefill cases against stage04. No single-chip result substitutes for the selected TP4 path.

Final supplemental dispositions:

- Reader microprofiles completed for both layer types: two readers wins all five projection roles, with exact-output PCC 1 and repeat spread at most 1.17%. The legal full-QK three-reader control loses as well. See `readers/*/performance.md`.
- Prefill trace trials pass repeated replay, but the modest sliding gain and absent full-attention gain do not justify an internal trace wrapper; capture overhead is explicitly recorded. The default remains composable into a caller-owned trace. See `profile_analysis.md`.
- Larger-batch prefill input-L1 advice is fully tested with role-specific block adaptations, actual allocator bounds, and two coherent streamed layout families across B8/B16/B32. All runnable adapted paths pass PCC but lose latency. See `input_l1_trials.md`, `input_l1_failures.json`, and `input_l1_capacity.json`.
- Eight final default profiles, 54 correctness cases, stack composition, allocation-tracked shared-workspace replay, context and worker watcher evidence are complete. `acceptance_completion.json` records the successful retry of the isolated profiler-marker failure without hiding the original queue exit.
