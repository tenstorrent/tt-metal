# Single-chip traffic and latency accounting

Nominal per-ASIC DRAM ceiling: 512 GB/s (`hardware_roofline_source.json`).
Only chip 0 is active. No 70%-roofline claim is made. The owner wall-cap stop
rule ends additional tuning without converting per-op estimates into a
whole-decoder utilization claim.

## Compulsory traffic and actual selected path

For one decoded row, top-10 routing requires the weights of 10 experts, not all
512 resident experts. BF16 gate/up/down compulsory weights are
`10 * (2560*640 + 2560*640 + 640*2560) * 2 = 98,304,000` bytes,
or 0.192 ms at the ideal DRAM ceiling. BFP8 uses 1088 bytes per 1024-element tile
including exponents: 52,224,000 bytes, or 0.102 ms. These are weight-only lower
bounds, not predicted decoder latency; activation/state traffic and repeated
reads caused by partitioning are separate.

The final path dispatches 10 indexed groups for physical count 1. Physical
count 2 uses the device union in at most 20 fixed slots; padding has zero scores
but still costs indexed sparse work. Counts 3..32 retain tile-union sparsity.
The final 32-row prefill profiles observe 104/156/185 active experts for layers
0/1/3. Packed gate/up merges dispatch/input work without reducing compulsory
expert weight bytes. PLE reads selected rows from its resident 30.72 GB table;
counting the whole table per token would inflate bandwidth utilization.
QSA reads/scoring depend on the actual prefix, so these 32-row profiles cannot
stand for maximum-context latency.

## Official sparse-row estimates

| Layer / phase | Expert dtype | Active | Gate/up ms | Down ms | Gate/up GB/s (% roof) | Down GB/s (% roof) |
|---|---|---:|---:|---:|---:|---:|
| 0 / prefill | bfloat16 | 104 | 7.700473 | 14.027295 | 91.83 (17.94%) | 25.81 (5.04%) |
| 0 / decode | bfloat16 | 10 | 0.702166 | 1.336647 | 96.83 (18.91%) | 26.05 (5.09%) |
| 1 / prefill | bfloat8_b | 156 | 11.342567 | 20.916203 | 48.45 (9.46%) | 13.75 (2.68%) |
| 1 / decode | bfloat8_b | 10 | 0.702607 | 1.331879 | 50.14 (9.79%) | 13.84 (2.70%) |
| 3 / prefill | bfloat16 | 185 | 13.372484 | 24.776730 | 94.06 (18.37%) | 26.00 (5.08%) |
| 3 / decode | bfloat16 | 10 | 0.702853 | 1.334694 | 96.74 (18.89%) | 26.09 (5.09%) |

These are official per-op estimates, not measured DRAM counters. Layer-1 BFP8
reduces estimated weight bytes but has similar decode sparse latency to BF16
layers; the resulting lower GB/s is not evidence of a wrong dtype. The CSV
explicitly records BFP8 operands and HiFi4. Physical-tile FLOP rates include
32 padded rows during one-row decode and must not be treated as logical-model
FLOP utilization.

## Same-capture decode accounting

| Layer | Device kernels ms | Device gaps ms | Device interval sum ms | Same capture host ms |
|---|---:|---:|---:|---:|---:|
| 0 | 4.966203 | 0.133837 | 5.100040 | 5.325551 |
| 1 | 6.290691 | 0.247841 | 6.538532 | 6.798739 |
| 3 | 9.404568 | 0.252566 | 9.657134 | 9.829245 |

Host values include replay submission/synchronization and profiler overhead.
They are not interchangeable with clean unprofiled ranking. The final clean
`wrap_final_rank_layer0.json`, `wrap_final_rank_layer1_final.json` and
`wrap_final_rank_layer3.json` report approximately 5.019, 6.341 and 9.548 ms
traced decode respectively. This separate regime is reported explicitly; no
difference is silently assigned to device compute or removed from the totals.

Prefill is segmented for per-op collection. Kernel sums are 78.003737,
92.397247 and 183.022748 ms, with gaps unmeasured. Clean warmed prefill is
approximately 78.400, 95.8 and 184.512 ms in the ranking artifacts. These are
separate runs: subtracting the segmented sum from clean host time does not
measure dispatch overhead. Historical whole-trace captures remain separately
labeled by runtime version and cannot replace final-source timing.

Exact inputs, advice and all operation families are in the six report
directories enumerated by `wrap_profile_summary.json`. Official advice for
L1 inputs, subblocks, K geometry, fidelity and DRAM sharding is carried under
`Not tried (stop rule)` in `per_op_findings.md`; no speculative improvement is
included in the measured results.
