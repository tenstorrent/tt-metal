# Final performance evidence

Warmed4096-token prefill and128 advancing teacher-forced decode steps; batch equals concurrency. Whole-layer device windows include all operations and internal gaps.

| Kind | B | Prefill before/after ms | Speedup | Decode before/after us | Speedup |
|---|---:|---:|---:|---:|---:|
| sliding_attention | 1 | 70.874 / 21.914 | 3.234x | 2996.815 / 958.869 | 3.125x |
| sliding_attention | 32 | 2201.643 / 688.938 | 3.196x | 4380.070 / 1932.889 | 2.266x |
| sliding_attention | 16 | 1131.259 / 337.279 | 3.354x | 3628.312 / 1414.217 | 2.566x |
| sliding_attention | 8 | 566.413 / 180.975 | 3.130x | 3213.684 / 1160.014 | 2.770x |
| full_attention | 1 | 99.991 / 31.643 | 3.160x | 3263.954 / 1074.807 | 3.037x |
| full_attention | 32 | 3198.766 / 991.363 | 3.227x | 6361.751 / 2855.908 | 2.228x |
| full_attention | 16 | 1600.111 / 490.729 | 3.261x | 4750.465 / 1934.582 | 2.456x |
| full_attention | 8 | 801.014 / 260.052 | 3.080x | 3947.907 / 1456.423 | 2.711x |

Before: ../agentic_serving.json fused_measurements and fused_decoder profiles. After: tracy_final/sliding_attention and tracy_final_k32/full_attention. Earlier tracy_final/full_attention used K128 and is superseded, not mixed into final numbers. Each directory contains advice-enabled prefill/decode tables and CSV, actual dtype/fidelity rows, complete128-replay checks, raw timestamps and same-run host timing.

## Device/host/roofline reconciliation

| Kind | B | Device decode us | Kernel sum us | Internal gaps us | Same-run host us without profiler drains |
|---|---:|---:|---:|---:|---:|
| sliding_attention | 1 | 958.869 | 925.839 | 33.030 | 986.349 |
| sliding_attention | 32 | 1932.889 | 1908.841 | 24.048 | 2102.253 |
| sliding_attention | 16 | 1414.217 | 1389.565 | 24.652 | 1506.289 |
| sliding_attention | 8 | 1160.014 | 1133.221 | 26.793 | 1211.788 |
| full_attention | 1 | 1074.807 | 1047.047 | 27.760 | 1103.849 |
| full_attention | 32 | 2855.908 | 2836.997 | 18.911 | 3177.595 |
| full_attention | 16 | 1934.582 | 1913.372 | 21.210 | 2101.400 |
| full_attention | 8 | 1456.423 | 1432.987 | 23.435 | 1551.238 |

Host includes harness teacher-forcing input/page-position copies and trace submissions; profiler drain time is separately recorded. These are not end-to-end token-out or serving metrics. Internal residual gaps remain in the denominator after packing, L1 carry, fused updates/multiply and projection padding. Required slice/reshard/head-layout boundaries are included, not hidden as external work. No claim is made that the remaining launch gaps are a proven generic runtime defect.

One participating ASIC theoretical peak:663.552TF/s LoFi and512GB/s. Useful FLOPs include dense projections and causal attention; estimated DRAM includes physical BFP headers, padding and active KV, excludes redundant core reads/cache reuse. These are estimates, not counters. No MoE inactive work exists. Mixed fidelity is divided by the whole-device LoFi peak, not averaged per-op utilization.

| Kind B1 | Useful prefill FLOPs | Prefill FLOPs % | Estimated decode DRAM bytes | Decode DRAM % |
|---|---:|---:|---:|---:|
| sliding_attention | 4043728486400 | 27.809 | 286016556 | 58.259 |
| full_attention | 4924314222592 | 23.453 | 328289452 | 59.656 |

## Short and tail prefill

Three warmed whole-prefill host trials; median shown. Includes every chunk and final synchronization, excludes input upload and first compilation. These are not device-time telemetry.

| Kind | Logical tokens | Physical chunks | Fused ms | Optimized ms |
|---|---:|---|---:|---:|
| sliding_attention | 33 | [64] | 3.027 | 3.575 |
| sliding_attention | 1024 | [1024] | 14.148 | 5.428 |
| sliding_attention | 1025 | [1024, 32] | 17.206 | 9.084 |
| sliding_attention | 2049 | [1024, 1024, 32] | 34.688 | 14.731 |
| sliding_attention | 4096 | [1024, 1024, 1024, 1024] | 74.793 | 22.562 |
| full_attention | 33 | [64] | 3.338 | 4.121 |
| full_attention | 1024 | [1024] | 17.425 | 6.901 |
| full_attention | 1025 | [1024, 32] | 20.733 | 11.041 |
| full_attention | 2049 | [1024, 1024, 32] | 43.697 | 18.893 |
| full_attention | 4096 | [1024, 1024, 1024, 1024] | 103.073 | 32.329 |

The selected4096-target policy is not a universal prefill win:33-token host prefill regresses by about0.548ms/18.1% sliding and0.783ms/23.4% full. The measured1024-token and longer boundary/tail cases improve substantially. This is an explicit short-workload tradeoff; no isolated root cause is inferred from whole-layer timing.

## Advice disposition

- Decode projections are marked optimized; explicit final BFP4/LoFi and two-reader rows are verified. One/two/three-reader microbenchmarks use the exact selected real tensor geometry; output-padding6144 supplies a legal three-reader alternative where needed and loses integrated timing.
- Missing prefill program_config advice is a reporter limitation: MinimalMatmulConfig is explicit under config=. M/K/N/subblock sweeps and legal2D alternatives are in minimal_geometry and prefill_refined.
- L1 input advice was implemented as private-input/in-place prefill at B1, with BF8 MLP intermediates. At B32 the1024-token BF16 input alone is352,321,536bytes, exceeding110*1.5MiB aggregate worker L1. The adapted32-token family below genuinely fits L1 and is measured against the default; this is not rejection at the first API error.
- Full B32 SDPA8x8 versus11x10 cannot increase active workers: sdpa_decode_program_factory.cpp:203–216 computes heads/core=ceil(4/(64//32))=2 or ceil(4/(110//32))=2, yielding64 workers either way. DRAM-Q avoids shard-grid restrictions; no unsupported parallelism gain is claimed.

| Kind B32 | Chunk tokens | Chunk count | Host prefill ms | Tail PCC versus default |
|---|---:|---:|---:|---:|
| sliding_attention | 1024 | 4 | 717.013 | 1.00000000 |
| sliding_attention | 32 | 128 | 886.711 | 0.99994147 |
| full_attention | 1024 | 4 | 1024.191 | 0.99999994 |
| full_attention | 32 | 128 | 1690.298 | 0.99998504 |
