# Roofline: nomic-embed-text-v2-moe on one Blackhole chip

8x512 on one p300c chip: the code of 2026-10-05 against its target at 70% compute and 60% DRAM
utilization.

| symbol | meaning |
|---|---|
| B | sequences per forward (batch size) |
| S | tokens per sequence |
| T | total token count, B x S; 8x512 is B = 8, S = 512, T = 4096 |
| H | hidden size, 768 |
| A | attention heads, 12 |
| F | FFN and expert intermediate size, 3072 |
| E | experts per MoE layer, 8; the router picks 2 per token (top-2), the code runs all 8 |

## Utilization and target

| 8x512 | compute | DRAM | bound | DRAM-bound below T | target, ms | current, ms | % of target |
|---|---|---|---|---|---|---|---|
| Dense layer | 42% | 11% | compute | 425 | 0.57 | 0.95 | 60% |
| MoE layer | 49% | 29% | compute | 689 | 2.31 | 3.30 | 70% |
| **e2e** | 47% | 25% | compute | 605 | 17.26 | 25.56 | 68% |

- **Utilization:** ideal time / time taken. Compute is FPU + SFPU work (overlapped for fc1 and w1,
  counter-measured for ops without FLOPs), DRAM runs at 512 GB/s.
- **Target:** max(compute / 0.70, DRAM / 0.60). % of target = target time / current time.
- **DRAM-bound below T:** below this token count, reading the weights outweighs the compute.

## Per op

8x512. every T: DRAM-bound at any token count; never: never DRAM-bound
| op | compute | DRAM | bound | DRAM-bound below T | % of target |
|---|---|---|---|---|---|
| embedding + norm | 6% | 63% | DRAM | every T | 106% |
| QKV | 60% | 16% | FPU | 581 | 86% |
| SDPA | 40% | 0% | FPU | never | 57% |
| out_proj | 45% | 5% | FPU | 464 | 65% |
| add + norm1 | 9% | 47% | DRAM | every T | 79% |
| fc1 + GELU | 57% | 8% | SFPU | 328 | 82% |
| fc2 | 62% | 6% | FPU | 463 | 89% |
| add + norm2 | 8% | 52% | DRAM | every T | 87% |
| expert w1 + GELU | 73% | 16% | SFPU | 201 | 104% |
| expert w2 | 43% | 51% | DRAM | every T | 85% |
| dispatch + combine | 2% | 66% | DRAM | every T | 110% |
