# Roofline: nomic-embed-text-v2-moe on one Blackhole chip

One p300c chip, the code of #59455 (small-input optimizations), against its target. Each op's target
is the time it takes at 70% compute utilization (60% for SDPA) or 60% DRAM bandwidth, whichever
limits it; the ops run one after another, as the model is built today, so a layer's and the
forward's target is the sum of its ops' targets. The target runs each token through its top-2
experts only; the code runs all 8. Every expert's weights are still read once a layer.

| symbol | meaning |
|---|---|
| B | sequences per forward (batch size) |
| S | tokens per sequence |
| T | total token count, B x S; 8x512 is B = 8, S = 512, T = 4096 |
| H | hidden size, 768 |
| A | attention heads, 12 |
| F | FFN and expert intermediate size, 3072 |
| E | experts per MoE layer, 8; the router picks 2 per token (top-2) |

## Target and current across T

![e2e device time, target and current, from 32 to 8192 tokens](roofline_e2e.svg)

## Layers and e2e
**Utilization:** ideal time / time taken. Compute is the target's FPU + SFPU work (overlapped for
  fc1 and w1, counter-measured for ops without FLOPs), with the experts top-2 routed; DRAM the bytes
  the code moves at 512 GB/s.
**Bound:** the bound of the ops that hold most of the target, with their share of it.
### 1x128: DRAM-bound

Bound by DRAM, reading the weights. Target: 60% DRAM bandwidth
| 1x128 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 8% | 20% | DRAM (98%) | 0.05 | 0.14 |
| MoE layer | 5% | 28% | DRAM (99%) | 0.16 | 0.33 |
| **e2e** | 6% | 25% | DRAM (99%) | 1.27 | 2.85 |

### 4x288: past the switch

The switch is a step at 802 tokens, where expert w1 + GELU turns compute-bound. Above it the norms,
expert w2 and dispatch + combine stay DRAM-bound

| 4x288 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 30% | 14% | compute (85%) | 0.18 | 0.36 |
| MoE layer | 12% | 27% | compute (52%) | 0.33 | 1.18 |
| **e2e** | 16% | 24% | compute (63%) | 3.07 | 9.28 |

### 8x256: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x256 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 39% | 13% | compute (85%) | 0.31 | 0.49 |
| MoE layer | 15% | 30% | compute (57%) | 0.54 | 1.70 |
| **e2e** | 20% | 26% | compute (67%) | 5.12 | 13.14 |

### 8x512: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x512 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 43% | 11% | compute (86%) | 0.66 | 0.93 |
| MoE layer | 16% | 28% | compute (62%) | 1.04 | 3.26 |
| **e2e** | 22% | 25% | compute (71%) | 10.29 | 25.20 |

## Per op

Every op as the code runs it: the ttnn call and its device time, per layer averaged over the 6 layers
of that type, with its share of the layer. Up to 320 tokens a MoE pass runs its
experts stacked (1x128): every expert's weights side by side in one matmul, the routing weights
applied before w2. Larger passes run transposed (4x288 and up): the tokens as columns, the routing
weights applied after w2.

### Embedding, once a forward

| op | ttnn call | 1x128 us | % of forward | 4x288 us | % of forward | 8x256 us | % of forward | 8x512 us | % of forward |
|---|---|---|---|---|---|---|---|---|---|
| token embedding lookup | `ttnn.embedding` | 6.2 | 0.2 | 12.7 | 0.1 | 18.4 | 0.1 | 31.6 | 0.1 |
| embedding norm | `ttnn.layer_norm` | 15.9 | 0.6 | 20.2 | 0.2 | 24.1 | 0.2 | 45.8 | 0.2 |

### Dense layer

| op | ttnn call | 1x128 us | % | 4x288 us | % | 8x256 us | % | 8x512 us | % |
|---|---|---|---|---|---|---|---|---|---|
| QKV projection | `ttnn.linear`; `ttnn.experimental.minimal_matmul` above 1024 tokens | 13.3 | **9.3** | 45.3 | **12.7** | 67.4 | **13.8** | 118.7 | **12.8** |
| head split | `ttnn.experimental.nlp_create_qkv_heads` | 13.2 | **9.2** | 19.6 | **5.5** | 22.7 | **4.7** | 36.2 | **3.9** |
| rotary on q and k | `ttnn.experimental.rotary_embedding_hf` x2 | 8.3 | **5.8** | 25.4 | **7.1** | 42.2 | **8.7** | 81.1 | **8.7** |
| attention | `ttnn.transformer.scaled_dot_product_attention` | 7.3 | **5.1** | 28.5 | **8.0** | 33.0 | **6.8** | 113.1 | **12.2** |
| head concat | `ttnn.experimental.nlp_concat_heads` | 5.2 | **3.6** | 6.2 | **1.7** | 7.1 | **1.5** | 11.6 | **1.2** |
| attention output projection | `ttnn.linear`; `minimal_matmul` above 1024 tokens | 8.6 | **6.0** | 24.0 | **6.7** | 31.0 | **6.4** | 52.4 | **5.6** |
| residual add + norm1 | `ttnn.layer_norm` | 19.3 | **13.5** | 22.5 | **6.3** | 27.6 | **5.7** | 51.8 | **5.6** |
| fc1 + GELU (FFN up-projection) | `ttnn.linear` + `ttnn.gelu`; one `ttnn.matmul` with the GELU fused above 1024 tokens | 22.8 | **15.9** | 82.3 | **23.0** | 140.0 | **28.8** | 259.4 | **27.9** |
| fc2 (FFN down-projection) | `ttnn.linear`; `minimal_matmul` above 1024 tokens | 25.5 | **17.8** | 81.6 | **22.8** | 87.9 | **18.1** | 153.9 | **16.5** |
| residual add + norm2 | `ttnn.layer_norm` | 19.6 | **13.7** | 22.6 | **6.3** | 27.8 | **5.7** | 52.4 | **5.6** |
| **layer total** | | 143.0 | **100** | 357.9 | **100** | 486.7 | **100** | 930.6 | **100** |

### MoE layer

| op | ttnn call | 1x128 us | % | 4x288 us | % | 8x256 us | % | 8x512 us | % |
|---|---|---|---|---|---|---|---|---|---|
| QKV projection | `ttnn.linear`; `ttnn.experimental.minimal_matmul` above 1024 tokens | 13.2 | **4.0** | 44.8 | **3.8** | 67.9 | **4.0** | 118.8 | **3.6** |
| head split | `ttnn.experimental.nlp_create_qkv_heads` | 13.2 | **4.0** | 19.7 | **1.7** | 22.6 | **1.3** | 35.6 | **1.1** |
| rotary on q and k | `ttnn.experimental.rotary_embedding_hf` x2 | 8.8 | **2.7** | 25.4 | **2.1** | 42.4 | **2.5** | 80.7 | **2.5** |
| attention | `ttnn.transformer.scaled_dot_product_attention` | 7.3 | **2.2** | 29.1 | **2.5** | 32.9 | **1.9** | 113.2 | **3.5** |
| head concat | `ttnn.experimental.nlp_concat_heads` | 5.2 | **1.6** | 6.1 | **0.5** | 7.2 | **0.4** | 11.5 | **0.4** |
| attention output projection | `ttnn.linear`; `minimal_matmul` above 1024 tokens | 8.6 | **2.6** | 23.9 | **2.0** | 30.8 | **1.8** | 52.4 | **1.6** |
| residual add + norm1 | `ttnn.layer_norm` | 19.3 | **5.9** | 22.6 | **1.9** | 27.5 | **1.6** | 51.9 | **1.6** |
| router scores | `ttnn.linear` + `ttnn.add` | 9.0 | **2.8** | 16.4 | **1.4** | 24.4 | **1.4** | 46.6 | **1.4** |
| router softmax + top-2 | `ttnn.softmax` + `ttnn.topk` | 21.9 | **6.7** | 24.1 | **2.0** | 23.3 | **1.4** | 35.7 | **1.1** |
| routing weights from the top-2 | `ttnn.slice`, `ttnn.typecast`, `ttnn.matmul`, `ttnn.eq`, `ttnn.matmul`, `ttnn.multiply` | 11.5 | **3.5** | 14.5 | **1.2** | 15.8 | **0.9** | 21.4 | **0.7** |
| x transposed (transposed pass) | `ttnn.transpose` | - | - | 10.0 | **0.8** | 16.4 | **1.0** | 34.1 | **1.0** |
| expert w1 + GELU, all 8 experts | `ttnn.matmul`, GELU fused | 101.0 | **30.8** | 583.2 | **49.3** | 838.5 | **49.5** | 1,629.2 | **50.0** |
| apply the routing weights | stacked: `ttnn.matmul`, `ttnn.multiply_`, `ttnn.reshard`; transposed: `ttnn.permute`, `ttnn.typecast`, `ttnn.multiply` | 18.4 | **5.6** | 64.0 | **5.4** | 113.1 | **6.7** | 209.9 | **6.4** |
| expert w2 | `ttnn.linear` with the bias, stacked; `ttnn.matmul` transposed | 71.9 | **21.9** | 214.9 | **18.2** | 316.4 | **18.7** | 585.6 | **18.0** |
| sum over experts + bias (transposed pass) | `ttnn.experimental.fast_reduce_nc`, `ttnn.transpose`, `ttnn.add` | - | - | 58.9 | **5.0** | 83.8 | **4.9** | 164.8 | **5.1** |
| residual add + norm2 | `ttnn.layer_norm` | 19.0 | **5.8** | 25.6 | **2.2** | 32.6 | **1.9** | 65.3 | **2.0** |
| **layer total** | | 328.4 | **100** | 1,183.0 | **100** | 1,695.6 | **100** | 3,256.8 | **100** |
