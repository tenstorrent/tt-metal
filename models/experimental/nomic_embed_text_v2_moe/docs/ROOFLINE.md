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
| Dense layer | 7% | 35% | DRAM (97%) | 0.05 | 0.10 |
| MoE layer | 4% | 51% | DRAM (99%) | 0.15 | 0.21 |
| **e2e** | 5% | 46% | DRAM (99%) | 1.17 | 1.88 |

### 4x288: past the switch

The forward turns compute-bound at 998 tokens, inside a band from 794 to 1256 where neither part holds
more than 56% of the target. The dense projections turn first (out_proj at 464 tokens, fc2 694, QKV and
fc1 + GELU 698), then the experts (w1 + GELU 1473, w2 2945); the embedding and dispatch + combine are
DRAM-bound at every T, SDPA and the norms compute-bound at every T.

| 4x288 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 42% | 22% | compute | 0.10 | 0.16 |
| MoE layer | 33% | 42% | DRAM (74%) | 0.18 | 0.26 |
| **e2e** | 36% | 34% | compute (51%) | 1.66 | 2.53 |

### 8x256: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x256 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 56% | 16% | compute | 0.17 | 0.21 |
| MoE layer | 46% | 34% | compute (71%) | 0.24 | 0.33 |
| **e2e** | 50% | 27% | compute (82%) | 2.47 | 3.23 |

### 8x512: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x512 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 66% | 9% | compute | 0.36 | 0.38 |
| MoE layer | 60% | 22% | compute (98%) | 0.47 | 0.52 |
| **e2e** | 62% | 17% | compute (98%) | 5.02 | 5.46 |

## Per op

Every op as the code runs it: the ttnn call and its device time, per layer averaged over the 6 layers
of that type, with its share of the layer. A MoE layer runs each token through its two routed
experts. From 1000 to 4096 tokens a pass, one generated op, `nomic_moe_route_dispatch`, scores the
tokens, picks each token's top-2, sorts the slots by expert and gathers the token rows into each
expert's region; one grouped matmul per projection multiplies each expert's rows by that expert's
weights; and the generated `nomic_moe_combine` weights and sums each token's two rows. Smaller passes
(1x128 here) run `router_front_end`, `routing_metadata` and `routed_combine` for the same steps. The
norms split the hidden axis over 8 core rows up to 128 tile rows (4096 tokens). From 3 tile rows a core
the ops around them read and write that layout, and only the embedding norm's input and the forward's
output convert; below it (1x128 here) every norm converts in and out.

### Once a forward: the embedding and the output

| op | ttnn call | 1x128 us | % of forward | 4x288 us | % of forward | 8x256 us | % of forward | 8x512 us | % of forward |
|---|---|---|---|---|---|---|---|---|---|
| token embedding lookup | `ttnn.embedding` | 6.2 | 0.3 | 10.7 | 0.4 | 14.9 | 0.5 | 22.1 | 0.4 |
| layout conversions around the embedding norm | `ttnn.to_memory_config` x2 (1x128); `Copy` (4x288, 8x256, 8x512) | 1.9 | 0.1 | 4.4 | 0.2 | 6.8 | 0.2 | 12.7 | 0.2 |
| embedding norm | `ttnn.layer_norm` (1x128); `ttnn.operations.nomic_residual_layer_norm (generated)` (4x288, 8x256, 8x512) | 7.0 | 0.4 | 7.4 | 0.3 | 9.1 | 0.3 | 14.1 | 0.3 |
| the last norm's output to interleaved DRAM | `ttnn.to_memory_config` (1x128); `Copy` (4x288, 8x256, 8x512) | 1.4 | 0.1 | 7.2 | 0.3 | 13.1 | 0.4 | 25.0 | 0.5 |

### Dense layer

| op | ttnn call | 1x128 us | % | 4x288 us | % | 8x256 us | % | 8x512 us | % |
|---|---|---|---|---|---|---|---|---|---|
| QKV projection | `ttnn.linear` | 12.7 | **13.1** | - | - | - | - | - | - |
| QKV + head split + rotary (one generated op) | `ttnn.operations.nomic_qkv_rotary_heads (generated)` | - | - | 36.0 | **23.0** | 45.9 | **22.1** | 83.0 | **22.0** |
| head split | `ttnn.experimental.nlp_create_qkv_heads` | 1.9 | **2.0** | - | - | - | - | - | - |
| rotary on q and k | `ttnn.experimental.rotary_embedding_llama` x2 | 6.7 | **6.9** | - | - | - | - | - | - |
| attention | `ttnn.operations.nomic_sdpa (generated)` | 4.3 | **4.4** | 15.6 | **9.9** | 19.8 | **9.5** | 52.9 | **14.0** |
| attention output projection | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 8.4 | **8.6** | 16.0 | **10.2** | 22.6 | **10.9** | 35.9 | **9.5** |
| residual add + norm1 | `ttnn.layer_norm` (1x128); `ttnn.operations.nomic_residual_layer_norm (generated)` (4x288, 8x256, 8x512) | 7.3 | **7.5** | 7.4 | **4.7** | 9.9 | **4.8** | 16.8 | **4.5** |
| fc1 + GELU (FFN up-projection) | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` | 20.1 | **20.6** | 32.3 | **20.6** | 48.3 | **23.3** | 86.4 | **22.9** |
| fc2 (FFN down-projection) | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 24.4 | **25.1** | 42.2 | **26.9** | 51.2 | **24.6** | 86.2 | **22.8** |
| residual add + norm2 | `ttnn.layer_norm` (1x128); `ttnn.operations.nomic_residual_layer_norm (generated)` (4x288, 8x256, 8x512) | 7.3 | **7.5** | 7.4 | **4.7** | 9.9 | **4.8** | 16.9 | **4.5** |
| layout conversions around the split norms | `ttnn.to_memory_config` x4 | 4.1 | **4.3** | - | - | - | - | - | - |
| **layer total** | | 97.4 | **100** | 156.9 | **100** | 207.7 | **100** | 378.1 | **100** |

### MoE layer

| op | ttnn call | 1x128 us | % | 4x288 us | % | 8x256 us | % | 8x512 us | % |
|---|---|---|---|---|---|---|---|---|---|
| QKV projection | `ttnn.linear` | 12.5 | **5.9** | - | - | - | - | - | - |
| QKV + head split + rotary (one generated op) | `ttnn.operations.nomic_qkv_rotary_heads (generated)` | - | - | 35.8 | **13.7** | 45.7 | **14.2** | 83.2 | **16.0** |
| head split | `ttnn.experimental.nlp_create_qkv_heads` | 1.9 | **0.9** | - | - | - | - | - | - |
| rotary on q and k | `ttnn.experimental.rotary_embedding_llama` x2 | 6.4 | **3.0** | - | - | - | - | - | - |
| attention | `ttnn.operations.nomic_sdpa (generated)` | 4.2 | **2.0** | 15.6 | **6.0** | 19.7 | **6.1** | 52.9 | **10.2** |
| attention output projection | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 8.4 | **3.9** | 16.0 | **6.2** | 22.6 | **7.0** | 35.8 | **6.9** |
| residual add + norm1 | `ttnn.layer_norm` (1x128); `ttnn.operations.nomic_residual_layer_norm (generated)` (4x288, 8x256, 8x512) | 7.3 | **3.4** | 7.4 | **2.8** | 9.9 | **3.1** | 16.8 | **3.2** |
| router scores + top-2 + routing metadata + dispatch (one generated op) | `ttnn.operations.nomic_moe_route_dispatch (generated)` | - | - | 15.7 | **6.0** | 20.1 | **6.2** | 31.5 | **6.0** |
| router scores | `ttnn.experimental.nomic_embed_text_v2_moe.router_front_end` | 8.5 | **4.0** | - | - | - | - | - | - |
| routing metadata: each token's top-2, each slot's row sorted by expert | `ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata` | 6.1 | **2.9** | - | - | - | - | - | - |
| dispatch: each token's row into its two experts' regions (gather mode) | `ttnn.experimental.nomic_embed_text_v2_moe.routed_combine` | 4.4 | **2.0** | - | - | - | - | - | - |
| expert w1 + GELU, each token's 2 experts | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` | 65.2 | **30.5** | 75.3 | **28.9** | 99.0 | **30.7** | 164.7 | **31.7** |
| expert w2 | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w2` (4x288, 8x256, 8x512) | 67.2 | **31.5** | 80.0 | **30.7** | 87.6 | **27.1** | 106.0 | **20.4** |
| combine: each token's two rows weighted and summed, plus the bias | `ttnn.experimental.nomic_embed_text_v2_moe.routed_combine` (1x128); `ttnn.operations.nomic_moe_combine (generated)` (4x288, 8x256, 8x512) | 10.1 | **4.7** | 7.0 | **2.7** | 8.3 | **2.6** | 12.7 | **2.4** |
| residual add + norm2 | `ttnn.layer_norm` (1x128); `ttnn.operations.nomic_residual_layer_norm (generated)` (4x288, 8x256, 8x512) | 8.1 | **3.8** | 7.4 | **2.8** | 9.9 | **3.1** | 16.8 | **3.2** |
| layout conversions around the split norms | `ttnn.to_memory_config` x3 | 3.3 | **1.5** | - | - | - | - | - | - |
| **layer total** | | 213.4 | **100** | 260.2 | **100** | 322.9 | **100** | 520.4 | **100** |
