# B3: TP=1 operation priorities, directly measured without topology extrapolation

**For the historical PP4 TP=1 stage, prioritize MoE Dispatch/Combine, SDPA, projection matmuls, and expert FFN. TP collectives are not a leading target at this shape.** These are investigation priorities from accumulated kernel work, not predicted throughput gains. A fresh standalone MLA sample subsequently passed on an eight-device Galaxy column; see the current-sample note below. LoudBox CI calibration remains separate.

## Evidence and exact metric

This report independently recomputes the committed `models/demos/deepseek_v3_d_p/tests/perf/captures/ops_{1rank,pp4_stage0,pp4_stage1,pp4_stage2,pp4_stage3}.csv.gz` files. Their README documents August 31, 2026 eager, instrumented execution on `bh-glx-110-a04u02`, one layer per PP stage, 5,120-token chunks growing KV context through eight chunks. These measurements are historical evidence, not a measurement of the September 10 traced implementation or a LoudBox CI sample.

The metric is **sum of non-socket device kernel durations / device count / nine SDPA invocations per device**. All durations are measured directly at their original topology; `approximate_mla_galaxy_perf` is never used. SHA-256 source hashes and exact operation values are in [b3_evidence.json](b3_evidence.json). The archive has nine SDPA invocations on every device, consistent with an additional invocation beyond the documented eight chunks. The reduced CSV lacks signposts/context metadata, so this reproduction retains all nine rather than guessing which row is warmup.

This is mean accumulated device work per invocation, **not layer elapsed time**, critical-path latency, steady throughput, TTFT, or a sum of per-operation device maxima. Durations of concurrent operations can overlap. Initial/setup operations present in the capture remain included and are amortized across the same nine invocations. Socket synchronization operations are excluded: their duration includes waiting for upstream work, so treating them as compute or transport cost is misleading.

The existing `analyze_prefill_layer_budget.py` describes grouping by global call count as a maximum across devices. In these archives the counts differ across device rows. Its published totals numerically reproduce the accumulated-work metric above; do not inherit its wall-time interpretation.

## TP=1 ranking

Stage 0 provides a concrete reference. Stages 1–2 show the sensitivity to layer identity; stage 3 additionally includes the LM head and final norm.

| Investigation target | Stage 0 ms | Stage 0 work share | Stages 0–2 ms range |
|---|---:|---:|---:|
| Dispatch + Combine | 4.239 | 35.65% | 4.239–6.950 |
| RingJointSDPA | 2.788 | 23.44% | 2.788–2.789 |
| All `MatmulDeviceOperation` instances | 2.085 | 17.53% | 2.083–2.085 |
| Routed expert FFN | 1.875 | 15.77% | 1.875–1.955 |
| LayerNorm | 0.155 | 1.31% | 0.154–0.155 |
| Residual AllGather | 0.020 | 0.16% | 0.020–0.021 |

Stage 0 total is **11.891 ms**; stages 1, 2, and 3 total **14.635, 11.971, and 16.171 ms**, respectively. These are accumulated-work budgets and must not be interpreted as stage service times. The stage-1 routing increase is also a different layer; it does not establish a placement problem or justify rebalancing the pipeline. The later 36-layer B1 analysis supports routing-operation variability within a fixed stage, while routing causality remains unproven.

Stage 3's matmul budget is **4.760 ms (29.44%)**, compared with approximately 2.084 ms in stages 0–2. This capture includes the LM head, so its ranking should not be copied onto ordinary interior layers or the KV-only final-layer profiling workload.

The TP=4 capture has **20.82%** in operations with `AllGather` or `ReduceScatter` in their names, including the distributed norm pre/post-all-gather operations. The same explicit category is **0.13–0.17%** across the four TP=1 stages, where only `AllGatherDeviceOperation` remains. This category is an operation-name budget, not a measurement of network utilization. Dispatch/Combine still communicate at TP=1; low residual collective cost does not mean communication disappeared. Direct Dispatch+Combine shares are 22.39% at TP=4 and 35.65% at TP=1 stage 0; the plan's rounded 26%→37% uses a broader routing categorization and must not be silently presented as this narrower pair.

## What to optimize and measure next

1. **MoE Dispatch/Combine:** B1 now demonstrates isolated Combine placement sensitivity in repeated local replays. Pair actual full-model PP routing and timings next; the historical gap's cause remains open. See [updated B1 finding](https://github.com/tenstorrent/tt-metal/blob/ssalice/mistral4-b1-investigation/profiling_reports/2026-09-11/B1_FINDING.md).
2. **SDPA and MLA movement:** benchmark the actual 8x1 shape across KV depths. The committed ramp's SDPA work grows strongly with context, so its position in the ranking depends on context. A query-dtype or movement change requires correctness and matched performance validation.
3. **Projection matmuls:** inspect the TP=1 shapes and memory/launch costs. The reduced archive merges all matmuls; it cannot isolate individual attention projections, routing projection, or LM head. Do not claim all matmul time is MLA time.
4. **Expert FFN:** compare observed expert loads and program configuration at TP=1 using the same routing/input. Its ~16% stage-0 share is meaningful but does not bound end-to-end savings without dependency information.

The legacy `MLA/attn` bucket includes SDPA, head creation/concatenation, rotary, KV-cache operations and softmax, but **excludes projection matmuls**. It is not the complete MLA block, and its number should not be compared directly with the new standalone MLA test's full measured operation range. Keep the standalone TP=1 MLA value in its original measured units; no galaxy approximation or calibrated gate is supplied by this historical analysis.

## Reproduce

From the `akhan/mistral4-prefill-followups` worktree:

```bash
python3 profiling_reports/2026-09-11/analyze_b3.py > /tmp/b3_evidence.json
cmp /tmp/b3_evidence.json profiling_reports/2026-09-11/b3_evidence.json
```

Completed successfully, device-free, using the Python standard library. [analyze_b3.py](analyze_b3.py) validates nonnegative durations, the device sets, and nine SDPA invocations on every device. It does not reconstruct a critical path from columns the archive does not contain. No issue was posted externally.

## Current standalone MLA sample, September 11

The new SP8/TP1 worker passed on `bh-glx-120-b10u14` using a topology-verified `TT_VISIBLE_DEVICES=0,1,2,3,11,10,9,8` ring and the recovered b48 native build plus the local B3 Python changes. At 51,200 cached tokens plus one 5,120-token chunk, it reported an 8.585052 ms merged operation budget: SDPA 7.195124 ms (83.81%), matmuls 0.737823 ms (8.59%), other 0.652105 ms (7.60%), and zero operations in the helper's collective category. This is a single functional/random-weight instrumented sample, not elapsed latency or a PCC result. It covers MLA only, so it cannot replace the historical full-layer ranking above. See [implementation and validation](B3_IMPLEMENTATION.md) and [run artifacts](b3-glx-column-PzSANj/).
