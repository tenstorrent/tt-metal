# Profiler accounting

All measurements use chip0 only. Nominal DRAM ceiling512GB/s is inherited from `../fused_decoder/hardware_roofline_source.json`. Useful work is active10 expert weight traffic, not all512resident experts or the whole PLE table. BFP8 expert compulsory weights are52,224,000bytes/token, ideal0.102ms; this excludes other projections, recurrent/cache traffic, repeats and control work.

## Same-capture decode

| Layer | Kernels ms | Device gaps ms | Kernel+gap ms | Profiled host replay ms | Clean default ms |
|---|---:|---:|---:|---:|---:|
| 0 | 2.985432 | 0.148143 | 3.133575 | 3.291117 | 3.051841 |
| 1 | 4.189593 | 0.257262 | 4.446855 | 4.672905 | 4.248298 |
| 3 | 7.161684 | 0.265737 | 7.427421 | 7.750205 | 7.296039 |

The profiled host column comes from that exact capture and includes instrumentation, submission and synchronization. The clean column is a separate regime and is never subtracted from profiler device time to infer a dispatch gap. Traced replay eliminates Python forward dispatch. Remaining generic/row-major boundaries are part of the decoder contract and counted.

## Physical prefill chunk32

| Layer | Kernel sum ms | Gate/up us | Down us | Active experts |
|---|---:|---:|---:|---:|
| 0 | 58.433279 | 1265.61 | 784.25 | 104 |
| 1 | 62.421907 | 1760.093 | 1020.565 | 158 |
| 3 | 147.872697 | 2056.916 | 1157.859 | 187 |

Prefill collection is segmented; inter-op gaps are unmeasured. Host durations inside segmented capture contain profiler drains, so clean whole-call prefill in the README is the latency authority. Logical tails pad internally and do not create a public32-token alignment requirement. Larger physical chunks/large-M2D exact-GR paths are deferred by the stop rule.

## Decode expert rows

| Layer | Gate/up us | Down us | Gate/up GB/s | Down GB/s |
|---|---:|---:|---:|---:|
| 0 | 105.719 | 49.747 | 309.95374530595257 | 329.3464932558747 |
| 1 | 104.828 | 49.884 | 312.588239783264 | 328.44198540614224 |
| 3 | 107.321 | 51.511 | 305.3270096253296 | 318.06798547882977 |

These bandwidth values are official report estimates, not hardware DRAM counters. All rows show BF16 activation/BFP8 weight/LoFi and active10. Larger output subblocks are tested by smaller-core gate alternatives and precision-locked final geometry. No full-model utilization is inferred; exact GR reductions and QSA indexing are not represented by an expert-only bandwidth bound.
