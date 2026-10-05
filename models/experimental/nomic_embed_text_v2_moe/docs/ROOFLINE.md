# Roofline: nomic-embed-text-v2-moe on one Blackhole chip

One p300c chip, the code of 2026-10-05, against its target at 70% compute utilization (60% for SDPA)
and 60% DRAM bandwidth.

| symbol | meaning |
|---|---|
| B | sequences per forward (batch size) |
| S | tokens per sequence |
| T | total token count, B x S; 8x512 is B = 8, S = 512, T = 4096 |
| H | hidden size, 768 |
| A | attention heads, 12 |
| F | FFN and expert intermediate size, 3072 |
| E | experts per MoE layer, 8; the router picks 2 per token (top-2), the code runs all 8 |

## Target and current across T

![e2e device time, target and current, from 32 to 8192 tokens](roofline_e2e.svg)

## Layers and e2e
**Utilization:** ideal time / time taken. Compute is FPU + SFPU work (overlapped for fc1 and w1,
  counter-measured for ops without FLOPs), DRAM the bytes the code moves at 512 GB/s.
### 1x128: DRAM-bound

Bound by DRAM, reading the weights. Target: 60% DRAM bandwidth
| 1x128 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 7% | 24% | DRAM | 0.05 | 0.17 |
| MoE layer | 12% | 22% | DRAM | 0.19 | 0.45 |
| **e2e** | 10% | 22% | DRAM | 1.45 | 3.75 |

### 2x288: the switch

Some parts compute, others DRAM bound

| 2x288 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 20% | 31% | compute | 0.08 | 0.27 |
| MoE layer | 30% | 30% | DRAM | 0.35 | 0.75 |
| **e2e** | 27% | 30% | DRAM | 2.48 | 6.17 |

### 8x256: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x256 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 38% | 13% | compute | 0.27 | 0.49 |
| MoE layer | 46% | 30% | compute | 1.14 | 1.72 |
| **e2e** | 44% | 27% | compute | 8.48 | 13.30 |

## Per op

\* DRAM-bound at every T.

### 1x128: DRAM-bound

| 1x128 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| embedding + norm\* | 1% | 8% | DRAM | 0.003 | 0.022 |
| QKV | 15% | 58% | DRAM | 0.146 | 0.179 |
| SDPA | 5% | 0% | FPU | 0.008 | 0.089 |
| out_proj | 8% | 31% | DRAM | 0.046 | 0.108 |
| add + norm1\* | 1% | 7% | DRAM | 0.015 | 0.246 |
| fc1 + GELU | 18% | 57% | DRAM | 0.096 | 0.154 |
| fc2 | 11% | 43% | DRAM | 0.092 | 0.157 |
| add + norm2\* | 1% | 6% | DRAM | 0.019 | 0.254 |
| expert w1 + GELU | 31% | 33% | DRAM | 0.461 | 0.727 |
| expert w2\* | 6% | 27% | DRAM | 0.473 | 0.855 |
| dispatch + combine\* | 17% | 11% | DRAM | 0.084 | 0.101 |

### 2x288: the switch

| 2x288 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| embedding + norm\* | 2% | 25% | DRAM | 0.012 | 0.028 |
| QKV | 30% | 42% | DRAM | 0.174 | 0.401 |
| SDPA | 19% | 0% | FPU | 0.075 | 0.234 |
| out_proj | 25% | 31% | FPU | 0.057 | 0.162 |
| add + norm1\* | 3% | 25% | DRAM | 0.069 | 0.260 |
| fc1 + GELU | 31% | 47% | SFPU | 0.180 | 0.411 |
| fc2 | 37% | 50% | FPU | 0.115 | 0.217 |
| add + norm2\* | 3% | 25% | DRAM | 0.086 | 0.259 |
| expert w1 + GELU | 53% | 22% | SFPU | 1.436 | 1.904 |
| expert w2\* | 25% | 54% | DRAM | 0.759 | 0.848 |
| dispatch + combine\* | 1% | 50% | DRAM | 0.377 | 0.452 |

### 8x256: compute-bound

| 8x256 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| embedding + norm\* | 5% | 58% | DRAM | 0.041 | 0.043 |
| QKV | 53% | 20% | FPU | 0.613 | 0.807 |
| SDPA | 36% | 0% | FPU | 0.238 | 0.399 |
| out_proj | 38% | 8% | FPU | 0.204 | 0.372 |
| add + norm1\* | 9% | 45% | DRAM | 0.246 | 0.331 |
| fc1 + GELU | 53% | 11% | SFPU | 0.638 | 0.840 |
| fc2 | 54% | 11% | FPU | 0.409 | 0.530 |
| add + norm2\* | 8% | 51% | DRAM | 0.307 | 0.363 |
| expert w1 + GELU | 71% | 18% | SFPU | 5.107 | 5.039 |
| expert w2\* | 39% | 52% | DRAM | 1.697 | 1.944 |
| dispatch + combine\* | 2% | 63% | DRAM | 1.341 | 1.281 |
