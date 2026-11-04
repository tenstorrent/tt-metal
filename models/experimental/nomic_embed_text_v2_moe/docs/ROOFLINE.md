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
| **e2e** | 5% | 45% | DRAM (99%) | 1.17 | 1.91 |

### 4x288: past the switch

The forward turns compute-bound at 998 tokens, inside a band from 794 to 1256 where neither part holds
more than 56% of the target. The dense projections turn first (out_proj at 464 tokens, fc2 694, QKV and
fc1 + GELU 698), then the experts (w1 + GELU 1473, w2 2945); the embedding and dispatch + combine are
DRAM-bound at every T, SDPA and the norms compute-bound at every T.

| 4x288 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 40% | 21% | compute | 0.10 | 0.16 |
| MoE layer | 31% | 40% | DRAM (74%) | 0.18 | 0.28 |
| **e2e** | 34% | 33% | compute (51%) | 1.66 | 2.67 |

### 8x256: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x256 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 55% | 16% | compute | 0.17 | 0.21 |
| MoE layer | 44% | 33% | compute (71%) | 0.24 | 0.34 |
| **e2e** | 48% | 27% | compute (82%) | 2.47 | 3.33 |

### 8x512: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x512 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 62% | 8% | compute | 0.36 | 0.40 |
| MoE layer | 57% | 21% | compute (98%) | 0.47 | 0.56 |
| **e2e** | 59% | 16% | compute (98%) | 5.02 | 5.80 |

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
| token embedding lookup | `ttnn.embedding` | 6.1 | 0.3 | 11.1 | 0.4 | 14.7 | 0.4 | 21.7 | 0.4 |
| layout conversions around the embedding norm | `ttnn.to_memory_config` x2 (1x128); `Copy` (4x288, 8x256, 8x512) | 1.9 | 0.1 | 4.2 | 0.2 | 6.5 | 0.2 | 12.5 | 0.2 |
| embedding norm | `ttnn.layer_norm` | 7.1 | 0.4 | 7.4 | 0.3 | 9.2 | 0.3 | 14.1 | 0.2 |
| the last norm's output to interleaved DRAM | `ttnn.to_memory_config` (1x128); `Copy` (4x288, 8x256, 8x512) | 1.4 | 0.1 | 7.8 | 0.3 | 12.8 | 0.4 | 24.4 | 0.4 |

### Dense layer

| op | ttnn call | 1x128 us | % | 4x288 us | % | 8x256 us | % | 8x512 us | % |
|---|---|---|---|---|---|---|---|---|---|
| QKV projection | `ttnn.linear` | 12.7 | **12.8** | - | - | - | - | - | - |
| QKV + head split + rotary (one generated op) | `ttnn.operations.nomic_qkv_rotary_heads (generated)` | - | - | 35.9 | **22.0** | 45.7 | **21.7** | 83.3 | **20.9** |
| head split | `ttnn.experimental.nlp_create_qkv_heads` | 1.9 | **2.0** | - | - | - | - | - | - |
| rotary on q and k | `ttnn.experimental.rotary_embedding_llama` x2 | 6.9 | **7.0** | - | - | - | - | - | - |
| attention | `ttnn.transformer.scaled_dot_product_attention` | 5.9 | **6.0** | 22.8 | **14.0** | 23.2 | **11.0** | 72.1 | **18.1** |
| attention output projection | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 8.3 | **8.4** | 15.9 | **9.7** | 22.7 | **10.7** | 37.4 | **9.4** |
| residual add + norm1 | `ttnn.layer_norm` | 7.4 | **7.4** | 7.5 | **4.6** | 10.0 | **4.7** | 16.9 | **4.2** |
| fc1 + GELU (FFN up-projection) | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` | 20.0 | **20.2** | 32.6 | **20.0** | 48.4 | **22.9** | 86.6 | **21.7** |
| fc2 (FFN down-projection) | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 24.4 | **24.6** | 41.4 | **25.3** | 51.1 | **24.2** | 86.4 | **21.6** |
| residual add + norm2 | `ttnn.layer_norm` | 7.4 | **7.5** | 7.4 | **4.5** | 10.1 | **4.8** | 16.8 | **4.2** |
| layout conversions around the split norms | `ttnn.to_memory_config` x4 | 4.2 | **4.2** | - | - | - | - | - | - |
| **layer total** | | 99.2 | **100** | 163.6 | **100** | 211.2 | **100** | 399.4 | **100** |

### MoE layer

| op | ttnn call | 1x128 us | % | 4x288 us | % | 8x256 us | % | 8x512 us | % |
|---|---|---|---|---|---|---|---|---|---|
| QKV projection | `ttnn.linear` | 12.4 | **5.7** | - | - | - | - | - | - |
| QKV + head split + rotary (one generated op) | `ttnn.operations.nomic_qkv_rotary_heads (generated)` | - | - | 35.9 | **13.0** | 45.8 | **13.6** | 82.9 | **14.9** |
| head split | `ttnn.experimental.nlp_create_qkv_heads` | 1.9 | **0.9** | - | - | - | - | - | - |
| rotary on q and k | `ttnn.experimental.rotary_embedding_llama` x2 | 6.4 | **2.9** | - | - | - | - | - | - |
| attention | `ttnn.transformer.scaled_dot_product_attention` | 5.9 | **2.7** | 22.8 | **8.2** | 23.3 | **6.9** | 72.1 | **13.0** |
| attention output projection | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 8.4 | **3.9** | 16.0 | **5.8** | 22.7 | **6.7** | 37.4 | **6.7** |
| residual add + norm1 | `ttnn.layer_norm` | 7.3 | **3.4** | 7.4 | **2.7** | 10.0 | **3.0** | 16.8 | **3.0** |
| router scores | `ttnn.experimental.nomic_embed_text_v2_moe.router_front_end` | 8.5 | **3.9** | 12.3 | **4.5** | 13.3 | **4.0** | 24.0 | **4.3** |
| routing metadata: each token's top-2, each slot's row sorted by expert | `ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata` | 6.1 | **2.8** | 7.7 | **2.8** | 10.7 | **3.2** | 12.4 | **2.2** |
| dispatch: each token's row into its two experts' regions (gather mode) | `ttnn.experimental.nomic_embed_text_v2_moe.routed_combine` | 4.3 | **2.0** | 4.1 | **1.5** | 5.3 | **1.6** | 9.0 | **1.6** |
| expert w1 + GELU, each token's 2 experts | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` | 66.0 | **30.5** | 75.9 | **27.4** | 99.3 | **29.4** | 164.9 | **29.7** |
| expert w2 | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w2` (4x288, 8x256, 8x512) | 67.5 | **31.2** | 79.7 | **28.8** | 88.4 | **26.2** | 106.1 | **19.1** |
| combine: each token's two rows weighted and summed, plus the bias | `ttnn.experimental.nomic_embed_text_v2_moe.routed_combine` | 10.2 | **4.7** | 7.4 | **2.7** | 8.5 | **2.5** | 13.0 | **2.3** |
| residual add + norm2 | `ttnn.layer_norm` | 8.1 | **3.7** | 7.5 | **2.7** | 10.0 | **3.0** | 16.8 | **3.0** |
| layout conversions around the split norms | `ttnn.to_memory_config` x3 | 3.3 | **1.5** | - | - | - | - | - | - |
| **layer total** | | 216.3 | **100** | 276.6 | **100** | 337.3 | **100** | 555.5 | **100** |
