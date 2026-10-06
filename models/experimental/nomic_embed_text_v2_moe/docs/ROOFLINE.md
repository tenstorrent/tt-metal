# Roofline: nomic-embed-text-v2-moe on one Blackhole chip

One p300c chip, the code of #59455 (small-input optimizations), against its target at 70% compute
utilization (60% for SDPA) and 60% DRAM bandwidth.

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
| Dense layer | 8% | 22% | DRAM | 0.05 | 0.14 |
| MoE layer | 15% | 28% | DRAM | 0.19 | 0.33 |
| **e2e** | 13% | 26% | DRAM | 1.44 | 2.87 |

### 2x288: the switch

Some parts compute, others DRAM bound

| 2x288 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 24% | 18% | compute | 0.08 | 0.22 |
| MoE layer | 31% | 29% | DRAM | 0.34 | 0.71 |
| **e2e** | 30% | 26% | DRAM | 2.45 | 5.62 |

### 8x256: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x256 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 39% | 13% | compute | 0.27 | 0.49 |
| MoE layer | 47% | 30% | compute | 1.14 | 1.70 |
| **e2e** | 45% | 26% | compute | 8.48 | 13.14 |

## Per op

\* DRAM-bound at every T.

### 1x128: DRAM-bound

| 1x128 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| embedding + norm\* | 1% | 8% | DRAM | 0.003 | 0.022 |
| QKV | 17% | 58% | DRAM | 0.146 | 0.158 |
| SDPA | 6% | 0% | FPU | 0.008 | 0.088 |
| out_proj | 9% | 28% | DRAM | 0.046 | 0.102 |
| add + norm1\* | 1% | 5% | DRAM | 0.015 | 0.237 |
| fc1 + GELU | 20% | 44% | DRAM | 0.096 | 0.137 |
| fc2 | 12% | 37% | DRAM | 0.092 | 0.152 |
| add + norm2\* | 1% | 6% | DRAM | 0.019 | 0.242 |
| expert w1 + GELU | 37% | 39% | DRAM | 0.461 | 0.607 |
| expert w2\* | 11% | 55% | DRAM | 0.473 | 0.433 |
| dispatch + combine\* | 8% | 9% | DRAM | 0.084 | 0.114 |

### 2x288: the switch

| 2x288 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| embedding + norm\* | 2% | 25% | DRAM | 0.012 | 0.028 |
| QKV | 40% | 36% | DRAM | 0.174 | 0.300 |
| SDPA | 19% | 0% | FPU | 0.075 | 0.233 |
| out_proj | 30% | 22% | FPU | 0.057 | 0.133 |
| add + norm1\* | 3% | 18% | DRAM | 0.069 | 0.249 |
| fc1 + GELU | 38% | 21% | SFPU | 0.180 | 0.330 |
| fc2 | 43% | 30% | FPU | 0.115 | 0.189 |
| add + norm2\* | 3% | 21% | DRAM | 0.086 | 0.253 |
| expert w1 + GELU | 53% | 22% | SFPU | 1.436 | 1.900 |
| expert w2\* | 25% | 54% | DRAM | 0.759 | 0.841 |
| dispatch + combine\* | 1% | 51% | DRAM | 0.377 | 0.448 |

### 8x256: compute-bound

| 8x256 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| embedding + norm\* | 5% | 58% | DRAM | 0.041 | 0.042 |
| QKV | 53% | 20% | FPU | 0.613 | 0.813 |
| SDPA | 36% | 0% | FPU | 0.238 | 0.393 |
| out_proj | 38% | 8% | FPU | 0.204 | 0.372 |
| add + norm1\* | 8% | 44% | DRAM | 0.246 | 0.333 |
| fc1 + GELU | 53% | 11% | SFPU | 0.638 | 0.840 |
| fc2 | 54% | 11% | FPU | 0.409 | 0.528 |
| add + norm2\* | 8% | 51% | DRAM | 0.307 | 0.363 |
| expert w1 + GELU | 71% | 18% | SFPU | 5.107 | 5.032 |
| expert w2\* | 40% | 54% | DRAM | 1.697 | 1.901 |
| dispatch + combine\* | 2% | 63% | DRAM | 1.341 | 1.272 |
