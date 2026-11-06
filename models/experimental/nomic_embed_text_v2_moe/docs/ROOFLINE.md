# Roofline: nomic-embed-text-v2-moe on one Blackhole chip

One p300c chip, the code of 2026-11-06 (v09), against the target of 2026-10-29, kept as it was. Each op's
target is the time it takes at 70% compute utilization (60% for SDPA) or 60% DRAM bandwidth,
whichever limits it; the ops run one after another, as the model is built today, so a layer's and the
forward's target is the sum of its ops' targets. The target was derived from the code of 2026-10-29,
whose fidelities and GELU the current code keeps: HiFi2 for QKV, fc1, fc2, the expert w1 and SDPA,
HiFi3 for out_proj, LoFi for the expert w2, HiFi4 for the router; grouped_matmul's polynomial GELU in
fc1 and the expert w1, 640 SFPU cycles a tile. It runs each token through its top-2 experts, reads
every weight once a layer, and moves the activation bytes that code moved per token at 8x512.

The current times are one `embed()` call on real text (SciFact abstracts cut to S, every token real), from the
token ids to the normalized embeddings: the backbone and the pooling tail (mean_pool as one matmul, the L2
normalize). Random token ids spread the tokens evenly over the experts, which real text does not, and the
tables before 2026-11-06 left the pooling out. The pooling's target is the final hidden state read once from
DRAM and the pooled vectors written once.

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
| **e2e** | 5% | 46% | DRAM (99%) | 1.17 | 1.87 |

### 4x288: past the switch

The forward turns compute-bound at 998 tokens, inside a band from 794 to 1256 where neither part holds
more than 56% of the target. The dense projections turn first (out_proj at 464 tokens, fc2 694, QKV and
fc1 + GELU 698), then the experts (w1 + GELU 1473, w2 2945); the embedding and dispatch + combine are
DRAM-bound at every T, SDPA and the norms compute-bound at every T.

| 4x288 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 42% | 22% | compute | 0.10 | 0.16 |
| MoE layer | 33% | 43% | DRAM (74%) | 0.18 | 0.26 |
| **e2e** | 36% | 35% | compute (51%) | 1.66 | 2.49 |

### 8x256: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x256 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 57% | 16% | compute | 0.17 | 0.21 |
| MoE layer | 48% | 35% | compute (71%) | 0.24 | 0.31 |
| **e2e** | 51% | 28% | compute (82%) | 2.48 | 3.13 |

### 8x512: compute-bound

Bound by compute. Target: 70% compute utilization (60% for SDPA)
| 8x512 | compute | DRAM | bound | target, ms | current, ms |
|---|---|---|---|---|---|
| Dense layer | 67% | 9% | compute | 0.36 | 0.37 |
| MoE layer | 63% | 22% | compute (98%) | 0.47 | 0.51 |
| **e2e** | 64% | 17% | compute (98%) | 5.04 | 5.33 |

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
| token embedding lookup | `ttnn.embedding` | 6.1 | 0.3 | 11.1 | 0.4 | 14.1 | 0.5 | 24.1 | 0.5 |
| layout conversions around the embedding norm | `ttnn.to_memory_config` x2 (1x128); `Copy` (4x288, 8x256, 8x512) | 1.8 | 0.1 | 4.6 | 0.2 | 7.0 | 0.2 | 11.6 | 0.2 |
| embedding norm | `ttnn.layer_norm` (1x128); `ttnn.operations.nomic_residual_layer_norm (generated)` (4x288, 8x256, 8x512) | 7.0 | 0.4 | 7.4 | 0.3 | 9.2 | 0.3 | 14.1 | 0.3 |
| the last norm's output to interleaved memory | `ttnn.to_memory_config` (1x128); `Copy` (4x288, 8x256, 8x512) | 1.4 | 0.1 | 3.8 | 0.2 | 6.3 | 0.2 | 11.2 | 0.2 |
| pooling: mean_pool (sum) + L2 normalize | `ttnn.matmul` + `ttnn.multiply` + `ttnn.sum` + `ttnn.clamp / ttnn.rsqrt` x2 + `ttnn.multiply` | 20.2 | 1.1 | 23.8 | 1.0 | 26.0 | 0.8 | 31.1 | 0.6 |

### Dense layer

| op | ttnn call | 1x128 us | % | 4x288 us | % | 8x256 us | % | 8x512 us | % |
|---|---|---|---|---|---|---|---|---|---|
| QKV projection | `ttnn.linear` | 12.4 | **12.9** | - | - | - | - | - | - |
| QKV + head split + rotary (one generated op) | `ttnn.operations.nomic_qkv_rotary_heads (generated)` | - | - | 35.9 | **23.0** | 45.9 | **22.3** | 83.2 | **22.2** |
| head split | `ttnn.experimental.nlp_create_qkv_heads` | 1.9 | **2.0** | - | - | - | - | - | - |
| rotary on q and k | `ttnn.experimental.rotary_embedding_llama` x2 | 6.4 | **6.6** | - | - | - | - | - | - |
| attention | `ttnn.operations.nomic_sdpa (generated)` | 4.3 | **4.5** | 15.7 | **10.1** | 19.9 | **9.7** | 52.5 | **14.0** |
| attention output projection | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 8.4 | **8.7** | 15.8 | **10.1** | 21.3 | **10.3** | 35.0 | **9.3** |
| residual add + norm1 | `ttnn.layer_norm` (1x128); `ttnn.operations.nomic_residual_layer_norm (generated)` (4x288, 8x256, 8x512) | 7.4 | **7.6** | 7.4 | **4.7** | 9.9 | **4.8** | 16.8 | **4.5** |
| fc1 + GELU (FFN up-projection) | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` | 20.1 | **20.8** | 32.4 | **20.8** | 48.5 | **23.6** | 86.6 | **23.1** |
| fc2 (FFN down-projection) | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 24.4 | **25.3** | 41.4 | **26.6** | 50.3 | **24.4** | 83.3 | **22.3** |
| residual add + norm2 | `ttnn.layer_norm` (1x128); `ttnn.operations.nomic_residual_layer_norm (generated)` (4x288, 8x256, 8x512) | 7.3 | **7.6** | 7.4 | **4.7** | 9.9 | **4.8** | 16.8 | **4.5** |
| layout conversions around the split norms | `ttnn.to_memory_config` x4 | 3.9 | **4.0** | - | - | - | - | - | - |
| **layer total** | | 96.5 | **100** | 155.9 | **100** | 205.7 | **100** | 374.1 | **100** |

### MoE layer

| op | ttnn call | 1x128 us | % | 4x288 us | % | 8x256 us | % | 8x512 us | % |
|---|---|---|---|---|---|---|---|---|---|
| QKV projection | `ttnn.linear` | 12.5 | **6.0** | - | - | - | - | - | - |
| QKV + head split + rotary (one generated op) | `ttnn.operations.nomic_qkv_rotary_heads (generated)` | - | - | 35.7 | **14.2** | 46.0 | **15.0** | 83.3 | **16.7** |
| head split | `ttnn.experimental.nlp_create_qkv_heads` | 1.9 | **0.9** | - | - | - | - | - | - |
| rotary on q and k | `ttnn.experimental.rotary_embedding_llama` x2 | 6.4 | **3.0** | - | - | - | - | - | - |
| attention | `ttnn.operations.nomic_sdpa (generated)` | 4.3 | **2.0** | 15.6 | **6.2** | 19.8 | **6.5** | 54.1 | **10.8** |
| attention output projection | `ttnn.linear` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.pirotski_matmul` (4x288, 8x256, 8x512) | 8.4 | **4.0** | 15.9 | **6.3** | 21.3 | **7.0** | 35.1 | **7.0** |
| residual add + norm1 | `ttnn.layer_norm` (1x128); `ttnn.operations.nomic_residual_layer_norm (generated)` (4x288, 8x256, 8x512) | 7.3 | **3.5** | 7.4 | **3.0** | 9.9 | **3.2** | 16.8 | **3.4** |
| router scores + top-2 + routing metadata + dispatch (one generated op) | `ttnn.operations.nomic_moe_route_dispatch (generated)` | - | - | 11.5 | **4.6** | 13.6 | **4.4** | 22.0 | **4.4** |
| router scores | `ttnn.experimental.nomic_embed_text_v2_moe.router_front_end` | 8.4 | **4.0** | - | - | - | - | - | - |
| routing metadata: each token's top-2, each slot's row sorted by expert | `ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata` | 6.3 | **3.0** | - | - | - | - | - | - |
| dispatch: each token's row into its two experts' regions (gather mode) | `ttnn.experimental.nomic_embed_text_v2_moe.routed_combine` | 4.4 | **2.1** | - | - | - | - | - | - |
| expert w1 + GELU, each token's 2 experts | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` | 62.8 | **29.9** | 69.9 | **27.9** | 91.0 | **29.7** | 160.3 | **32.1** |
| expert w2 | `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w1` (1x128); `ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul_w2` (4x288, 8x256, 8x512) | 66.2 | **31.5** | 79.3 | **31.7** | 86.3 | **28.2** | 98.4 | **19.7** |
| combine: each token's two rows weighted and summed, plus the bias | `ttnn.experimental.nomic_embed_text_v2_moe.routed_combine` (1x128); `ttnn.operations.nomic_moe_combine (generated)` (4x288, 8x256, 8x512) | 10.0 | **4.8** | 7.8 | **3.1** | 8.2 | **2.7** | 12.6 | **2.5** |
| residual add + norm2 | `ttnn.layer_norm` (1x128); `ttnn.operations.nomic_residual_layer_norm (generated)` (4x288, 8x256, 8x512) | 8.0 | **3.8** | 7.5 | **3.0** | 9.9 | **3.2** | 16.9 | **3.4** |
| layout conversions around the split norms | `ttnn.to_memory_config` x3 | 3.2 | **1.5** | - | - | - | - | - | - |
| **layer total** | | 209.9 | **100** | 250.6 | **100** | 306.1 | **100** | 499.6 | **100** |
