# Roofline: nomic-embed-text-v2-moe on one Blackhole chip

One p300c chip, the code of 2026-11-03, against the target of 2026-10-29, kept as it was. Each op's
target is the time it takes at 70% compute utilization (60% for SDPA) or 60% DRAM bandwidth,
whichever limits it; the ops run one after another, as the model is built today, so a layer's and the
forward's target is the sum of its ops' targets. The target was derived from the code of 2026-10-29,
whose fidelities and GELU the current code keeps: HiFi2 for QKV, fc1, fc2, the expert w1 and SDPA,
HiFi3 for out_proj, LoFi for the expert w2, HiFi4 for the router; grouped_matmul's polynomial GELU in
fc1 and the expert w1, 640 SFPU cycles a tile. It runs each token through its top-2 experts, reads
every weight once a layer, and moves the activation bytes that code moved per token at 8x512.

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

![e2e device time, target and current, from 32 to 16384 tokens](roofline_e2e.svg)

## Layers and e2e
**Utilization:** ideal time / time taken. Compute is the target's FPU + SFPU work (overlapped for
  fc1 and w1, counter-measured for ops without FLOPs), with the experts top-2 routed; DRAM the bytes
  the code moves at 512 GB/s.
**Bound:** the bound of the ops that hold most of the target, with their share of it.
### 1x128: DRAM-bound

Bound by DRAM, reading the weights. Target: 60% DRAM bandwidth
| 1x128 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 7% | 34% | DRAM (97%) | 0.05 | 0.10 |
| MoE layer | 4% | 50% | DRAM (99%) | 0.15 | 0.22 |
| **e2e** | 5% | 45% | DRAM (99%) | 1.17 | 1.90 |

### 4x288: past the switch

The forward turns compute-bound at 998 tokens, inside a band from 794 to 1256 where neither part holds
more than 56% of the target. The dense projections turn first (out_proj at 464 tokens, fc2 694, QKV and
fc1 + GELU 698), then the experts (w1 + GELU 1473, w2 2945); the embedding and dispatch + combine are
DRAM-bound at every T, SDPA and the norms compute-bound at every T.

| 4x288 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 40% | 21% | compute | 0.10 | 0.16 |
| MoE layer | 29% | 38% | DRAM (74%) | 0.18 | 0.29 |
| **e2e** | 33% | 32% | compute (51%) | 1.66 | 2.75 |

### 8x256: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x256 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 55% | 16% | compute | 0.17 | 0.21 |
| MoE layer | 42% | 31% | compute (71%) | 0.24 | 0.36 |
| **e2e** | 46% | 26% | compute (82%) | 2.47 | 3.46 |

### 8x512: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x512 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 63% | 8% | compute | 0.36 | 0.40 |
| MoE layer | 54% | 19% | compute (98%) | 0.47 | 0.59 |
| **e2e** | 57% | 15% | compute (98%) | 5.02 | 5.98 |

## Per op

Every op as the code runs it: the ttnn call and its device time, per layer averaged over the 6 layers
of that type, with its share of the layer. A MoE layer runs each token through its two routed
experts: `routing_metadata` sorts the token rows by expert on the device, `routed_combine` gathers
them into each expert's region, one grouped matmul per projection multiplies each expert's rows by
that expert's weights, and `routed_combine` weights and sums each token's two rows. The norms split
the hidden axis over 8 core rows up to 128 tile rows (4096 tokens). From 3 tile rows a core the ops
around them read and write that layout, and only the embedding norm, the router's input and the
forward's output convert; below it (1x128 here) every norm converts in and out.

### Once a forward: the embedding and the output

| op | ttnn call | 1x128 us | % of forward | 4x288 us | % of forward | 8x256 us | % of forward | 8x512 us | % of forward |
|---|---|---|---|---|---|---|---|---|---|
| token embedding lookup | `ttnn.embedding` | 6.2 | 0.3 | 11.0 | 0.4 | 14.7 | 0.4 | 21.9 | 0.4 |
| layout conversions around the embedding norm | `ttnn.to_memory_config` x2 (1x128); `Copy` (4x288, 8x256, 8x512) | 1.9 | 0.1 | 4.3 | 0.2 | 6.5 | 0.2 | 12.4 | 0.2 |
| embedding norm | `ttnn.layer_norm` | 7.1 | 0.4 | 7.3 | 0.3 | 9.2 | 0.3 | 14.1 | 0.2 |
| the last norm's output to interleaved DRAM | `ttnn.to_memory_config` (1x128); `Copy` (4x288, 8x256, 8x512) | 1.6 | 0.1 | 7.2 | 0.3 | 12.8 | 0.4 | 25.6 | 0.4 |

### Dense layer

| op | ttnn call | 1x128 us | % | 4x288 us | % | 8x256 us | % | 8x512 us | % |
|---|---|---|---|---|---|---|---|---|---|
| QKV projection | `ttnn.linear` | 12.6 | **12.7** | - | - | - | - | - | - |
| QKV + head split + rotary (one generated op) | `ttnn.operations.nomic_qkv_rotary_heads (generated)` | - | - | 36.0 | **21.9** | 45.9 | **21.7** | 82.9 | **20.8** |
| head split | `ttnn.experimental.nlp_create_qkv_heads` | 2.0 | **2.0** | - | - | - | - | - | - |
| rotary on q and k | `ttnn.experimental.rotary_embedding_llama` x2 | 6.6 | **6.7** | - | - | - | - | - | - |
| attention | `ttnn.transformer.scaled_dot_product_attention` | 5.9 | **6.0** | 22.8 | **13.9** | 23.3 | **11.0** | 72.2 | **18.1** |
| attention output projection | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 8.4 | **8.5** | 16.0 | **9.7** | 22.6 | **10.7** | 37.3 | **9.3** |
| residual add + norm1 | `ttnn.layer_norm` | 7.4 | **7.5** | 7.4 | **4.5** | 10.0 | **4.7** | 16.9 | **4.2** |
| fc1 + GELU (FFN up-projection) | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` | 19.9 | **20.2** | 32.4 | **19.7** | 48.4 | **22.9** | 86.5 | **21.7** |
| fc2 (FFN down-projection) | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 24.4 | **24.7** | 42.3 | **25.7** | 51.0 | **24.2** | 86.1 | **21.6** |
| residual add + norm2 | `ttnn.layer_norm` | 7.4 | **7.5** | 7.5 | **4.5** | 10.0 | **4.7** | 16.8 | **4.2** |
| layout conversions around the split norms | `ttnn.to_memory_config` x4 | 4.2 | **4.2** | - | - | - | - | - | - |
| **layer total** | | 98.7 | **100** | 164.2 | **100** | 211.2 | **100** | 398.7 | **100** |

### MoE layer

| op | ttnn call | 1x128 us | % | 4x288 us | % | 8x256 us | % | 8x512 us | % |
|---|---|---|---|---|---|---|---|---|---|
| QKV projection | `ttnn.linear` | 12.5 | **5.8** | - | - | - | - | - | - |
| QKV + head split + rotary (one generated op) | `ttnn.operations.nomic_qkv_rotary_heads (generated)` | - | - | 35.9 | **12.5** | 46.0 | **12.8** | 83.1 | **14.2** |
| head split | `ttnn.experimental.nlp_create_qkv_heads` | 1.9 | **0.9** | - | - | - | - | - | - |
| rotary on q and k | `ttnn.experimental.rotary_embedding_llama` x2 | 6.3 | **2.9** | - | - | - | - | - | - |
| attention | `ttnn.transformer.scaled_dot_product_attention` | 5.9 | **2.7** | 22.8 | **7.9** | 23.4 | **6.5** | 72.1 | **12.3** |
| attention output projection | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 8.4 | **3.9** | 15.9 | **5.5** | 22.6 | **6.3** | 37.5 | **6.4** |
| residual add + norm1 | `ttnn.layer_norm` | 7.3 | **3.4** | 7.5 | **2.6** | 10.0 | **2.8** | 16.9 | **2.9** |
| router scores | `ttnn.experimental.nomic_embed_text_v2_moe.router_front_end` | 8.5 | **3.9** | 12.3 | **4.3** | 13.1 | **3.7** | 24.0 | **4.1** |
| routing metadata: each token's top-2, each slot's row sorted by expert | `ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata` | 6.1 | **2.8** | 7.6 | **2.6** | 10.7 | **3.0** | 12.5 | **2.1** |
| dispatch: each token's row into its two experts' regions (gather mode) | `ttnn.experimental.nomic_embed_text_v2_moe.routed_combine` | 4.4 | **2.0** | 10.0 | **3.5** | 12.9 | **3.6** | 21.7 | **3.7** |
| expert w1 + GELU, each token's 2 experts | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` | 64.8 | **30.0** | 75.7 | **26.2** | 101.2 | **28.2** | 164.0 | **28.0** |
| expert w2 | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w2` (4x288, 8x256, 8x512) | 68.2 | **31.6** | 80.0 | **27.7** | 89.7 | **25.0** | 106.4 | **18.2** |
| combine: each token's two rows weighted and summed, plus the bias | `ttnn.experimental.nomic_embed_text_v2_moe.routed_combine` | 10.2 | **4.7** | 13.2 | **4.6** | 19.1 | **5.3** | 30.6 | **5.2** |
| residual add + norm2 | `ttnn.layer_norm` | 8.1 | **3.8** | 7.4 | **2.6** | 10.0 | **2.8** | 16.8 | **2.9** |
| layout conversions around the split norms | `ttnn.to_memory_config` x3 | 3.3 | **1.6** | - | - | - | - | - | - |
| **layer total** | | 215.9 | **100** | 288.4 | **100** | 358.7 | **100** | 585.5 | **100** |
