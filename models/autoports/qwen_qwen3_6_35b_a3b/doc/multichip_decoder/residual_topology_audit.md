# Residual Topology Audit

This audit records the alternatives considered before keeping the decoder
layer-boundary residual stream replicated on the local `2x2` Blackhole mesh.
All hidden tensors are BF16, so full hidden is `2048 * 2 = 4096` bytes per
token and a TP-local half-hidden shard is `1024 * 2 = 2048` bytes per token.

| Case | Replicated bytes | TP-local bytes |
| --- | ---: | ---: |
| Decode token, batch 1 | 4096 B | 2048 B |
| Linear prefill seq 5, batch 1 | 20480 B | 10240 B |
| Full prefill seq 33, batch 1 | 135168 B | 67584 B |

The source layer contract stores a residual, applies full-hidden RMSNorm,
adds token-mixer output, applies post-attention full-hidden RMSNorm, runs MoE,
and then adds the full-hidden residual. The next layer consumes the same
full-hidden shape. Current multichip projection weights shard output/expert
dimensions but keep `K=2048` as the input dimension.

| Alternative | Residual layout | Next consumer | CCL dtype and bytes | Persistent-buffer plan | Result or exact blocker |
| --- | --- | --- | --- | --- | --- |
| Replicated residual plus `ttnn.all_reduce` after row-parallel outputs | `[1,batch,seq,2048]` on every device | Existing residual adds, full-hidden RMSNorm, then QKV/linear/MoE projections | BF16 full-hidden output. Final perf reports show `ttnn.all_reduce` materialized as `ReduceScatterDeviceOperation` plus `AllGatherDeviceOperation`: linear prefill CCL 0.265 ms, linear decode 0.087 ms, full prefill 1.530 ms, full decode 0.086 ms. | No extra persistent hidden buffer. KV cache and linear recurrent/conv state remain TP-sharded; layer output is replicated for the future stack. | Selected and validated by PCC, trace replay, fallback audit, watcher run, and perf reports. |
| Reduce-scatter only at token-mixer output with delayed all-gather | TP-local `[1,batch,seq,1024]` after attention/linear output | Immediate `ttnn.add(residual, mixer_out)`, then `_rms_norm(..., weight=[1,1,1,2048])` | Would communicate only the TP shard initially: 2 KiB per decode token, 10 KiB for seq5, 66 KiB for seq33, BF16. | Requires persistent sharded residual plus a gathered full-hidden scratch before the next full-hidden op. | Blocked by shape contract: the residual add and RMSNorm expect full hidden. Delaying all-gather only until the next line still requires a distributed RMSNorm rewrite or an immediate all-gather back to `[... ,2048]`. |
| Reduce-scatter only at MoE output with delayed all-gather to next layer | TP-local `[1,batch,seq,1024]` layer output | Next decoder layer input RMSNorm and QKV/linear/MoE projections | Would keep a 2 KiB/token TP-local layer-boundary shard, BF16. | Full model stack would need persistent sharded layer-boundary buffers and a gather scratch before every next-layer full-hidden projection. | Blocked by stack contract: next layer `input_layernorm` and every current input projection consume full hidden `K=2048`. The gather moves to the next layer boundary rather than being removed. |
| Fused all-gather plus input matmul | TP-local residual shard, gathered inside first projection | QKV/linear packed projections and MoE router/shared/routed projections | Same BF16 full-hidden payload as all-gather, but fused with projection launch. | Would keep sharded residual persistently and avoid a separately materialized gathered hidden only if the fused op owns the gather scratch. | Not available in this stage's TTNN path. Current `ttnn.linear` calls are standard full-K matmuls, and the loaded weights are sharded on output/expert axes while the input K dimension remains 2048. |
| Fused matmul plus reduce-scatter output | Replicated input, TP-local output shard | Residual add/RMSNorm if kept sharded, or all-gather before residual add | BF16 reduce-scatter payload is half-hidden per TP column. | Would need sharded residual/output buffers plus either distributed residual/RMSNorm or gather scratch. | Not a completion path by itself: it only changes the final reduction output. The immediate residual add and RMSNorm still need full-hidden values unless those ops are also rewritten. |
| Fully sharded residual stream with distributed RMSNorm | TP-local `[... ,1024]` across the full layer stack | Distributed RMSNorm, then projections that accept sharded K or fused gather | BF16 collectives for RMS sum-of-squares plus projection gathers/reductions. | Persistent sharded boundary buffers for every layer; full hidden is materialized only inside fused projection/RMS helpers. | Out of scope for this decoder-stage baseline. It requires new distributed RMSNorm semantics and projection kernels whose public contract differs from the completed optimized decoder. |
| 2D residual sharding across TP and EP | Quarter hidden or token/expert-sharded residual | RMSNorm, router, MoE, and next layer projections | BF16 payload may shrink per device but adds EP synchronization for non-expert ops. | Persistent 2D hidden buffers plus gather/scatter bridges around all non-EP consumers. | Rejected: EP rows do not own hidden partitions in the implemented math. EP is used only to split gate-selected active MoE execution; all non-MoE ops are TP/replicated. |

MoE-specific note: the completed path uses `moe_routing_remap` to distribute
the selected top-k sparse rows across EP rows. Expert weights remain replicated
across EP rows and there is no fixed contiguous expert-range ownership. Broad
multi-token active sparse prefill is rejected because `moe_routing_remap`
accepts one routing row `[1,E]`, and the no-remap token-by-expert sparse probe
recorded in `triage/active_prefill_sparse_probe_*` hung in
`SparseMatmulDeviceOperation`.

Generated audit data is in `logs/residual_topology_audit.log`.
