# Final device-profile inspection

Times below are per-device sums of kernel durations, averaged across four devices. They diagnose cost categories; they are not critical-path times or roofline denominators. Whole-layer firmware envelopes in performance.json include all device gaps.

| Layer | Phase | Projection | CCL | Movement | Other | Whole layer |
|---|---|---:|---:|---:|---:|---:|
| linear_attention | prefill | 2431.46 µs | 2706.54 µs | 913.57 µs | 2026.33 µs | 8326.15 µs |
| linear_attention | decode | 175.76 µs | 64.60 µs | 96.82 µs | 114.17 µs | 508.85 µs |
| full_attention | prefill | 2719.25 µs | 2707.50 µs | 990.14 µs | 4009.37 µs | 10628.84 µs |
| full_attention | decode | 172.42 µs | 64.93 µs | 38.86 µs | 106.40 µs | 424.10 µs |

Actual AG+RS communication costs about 2.71 ms per prefill layer and65µs per decode layer. Distributed RMSNorm's local pre/post-statistics kernels are counted under Other, not CCL. Communication is a major remaining prefill cost; the sharded residual contract avoids immediate replication but still requires input gathers at column projections. Decode projection kernels dominate the remaining local work, while linear attention also pays recurrence and substantial reshape/slice traffic. Full prefill includes about2.57ms of SDPA kernel time. These category sums are diagnostic and do not replace the complete firmware envelope.

The human tables flag prefill matmuls and the small decode output projection as SLOW. Decode smaller-core/larger-K geometry was measured at the same BFP4/LoFi policy (`geometry_results.json`, `extra_geometry_results.json`); corresponding prefill geometry is measured in `prefill_geometry_results.json`. Packing gate/up removes repeated input reads and separate weight storage. Fused CCL families were adapted to the sharded consumer contract and measured, then rejected as slower. Generic advice to increase fidelity is not followed: the preserved optimized baseline is BFP4/LoFi and direct model-weight PCC passes.

Per-op DRAM/utilization estimates in tt-perf-report are diagnostics. Final whole-layer rooflines are14.31%/11.55% useful prefill FLOPs and21.85%/25.18% estimated decode DRAM bandwidth for linear/full attention. The denominator includes all layer operations and gaps; decode traffic is a lower bound excluding intermediate spills, CCL staging and rereads. Physical peaks use120cores/ASIC at1.35GHz and512GB/s/ASIC across all four ASICs.

The completed precision-locked prefill geometry screen found no reproducible whole-layer improvement from hiddenK20, mixedK40, outputK24, downK34 or smaller11×8/8×8 grids. Twenty-replay full-attention confirmation retains the control. See prefill_selection.json for distributions; the SLOW labels are investigated, not treated as evidence that an untested geometry is impossible.
