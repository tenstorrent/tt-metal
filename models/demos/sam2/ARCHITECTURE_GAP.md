# Phase 0: Honest Architecture-Difference Table
## HF Sam2Model vs Our TTNN Implementation

| Component | Official HF Implementation | Our TTNN Implementation | Status |
|-----------|---------------------------|------------------------|--------|
| **Image Encoder Backbone** | | | |
| Patch Embedding | `Conv2d(3, 96, kernel=7, stride=4, padding=3)` | `ttnn.linear` (wrong op) | **WRONG** |
| Positional Embedding | `Sam2PositionalEmbedding` — sine-based spatial | None | **MISSING** |
| Window Positional Embedding | `window_positional_embedding_background` per stage | None | **MISSING** |
| Hiera Blocks | 12 `Sam2MultiScaleBlock` with window/global attention | 4 simple SDPA blocks | **WRONG** |
| Block count per stage | Blocks per stage: [1, 2, 7, 2] = 12 total (config) | 1 per stage = 4 total | **WRONG** |
| Window sizes per stage | [8, 4, 14, 7] | None (full SDPA) | **WRONG** |
| Global attention blocks | Blocks [5, 7, 9] (indices, from config) | None | **MISSING** |
| LayerNorm | `layer_norm1` before attn, `layer_norm2` before MLP | None | **MISSING** |
| QKV projection | `qkv` Linear per block with 3*C*dim output | Separate ttnn.linear with same weight | **WRONG** |
| Multi-scale attention | Windowed + `Sam2MultiScaleAttention` with pooling | Simple SDPA | **WRONG** |
| MLP | `Sam2FeedForward` — 2 Linear + GELU | None | **MISSING** |
| Residual connections | Shortcut → add → norm | None | **MISSING** |
| Stage transitions | `proj` Linear per block + query pooling | `avg_pool2d` (host fallback) | **WRONG** |
| **Neck/FPN** | | | |
| Feature levels | 3 levels: [256,256], [128,128], [64,64] (config) | 4 levels output | **WRONG** |
| FPN layers | `Sam2VisionNeck` with conv+upsample for FPN | None | **MISSING** |
| FPN hidden_size | 256 (config) | N/A | **MISSING** |
| **Prompt Encoder** | | | |
| Point encoding | Positional encoding + cos/sin + `point_embed` (4 types) | `ttnn.linear` direct projection | **WRONG** |
| Point embeddings | `point_embed.weight` — 4 learned [1, 256] (pos/neg/box_coords) | Random `torch.randn` | **WRONG** |
| Not-a-point | `not_a_point_embed.weight` — learned padding embedding | `torch.zeros` fallback | **WRONG** |
| No-mask embedding | `no_mask_embed.weight` — learned [1, 256] | None | **MISSING** |
| Mask downscaling | `mask_embed`: 3x Conv2d(k=3,s=2,p=1) + 2x LayerNorm + GELU | None | **MISSING** |
| Coordinate encoding | Shared positional embedding + scaling | None | **MISSING** |
| **Mask Decoder** | | | |
| Learned tokens | `iou_token` [1,256], `mask_tokens` [4,256], `obj_score_token` [1,256] | None | **MISSING** |
| Two-way transformer | 2x `Sam2TwoWayAttentionBlock` | 1 simple SDPA cross-attention | **WRONG** |
| Self-attention | `self_attn` (q_proj, k_proj, v_proj, o_proj) in each block | None | **MISSING** |
| Cross-attn token→image | `cross_attn_token_to_image` | None | **MISSING** |
| Cross-attn image→token | `cross_attn_image_to_token` | None | **MISSING** |
| LayerNorm x4 per block | 4 layernorms (before each attn/mlp/cross-attn) | None | **MISSING** |
| MLP per block | `Sam2FeedForward` with ReLU (proj_in 2048, proj_out 256) | None | **MISSING** |
| Final attn token→image | `final_attn_token_to_image` after transformer layers | None | **MISSING** |
| Final LayerNorm | `layer_norm_final_attn` | None | **MISSING** |
| Upscaling | `upscale_conv1` ConvTranspose2d, `upscale_conv2` ConvTranspose2d, GELU, `upscale_layer_norm` | Direct linear to [1,256,256] | **WRONG** |
| Output hypernetworks | 4x `output_hypernetworks_mlps` (ReLU MLP: 256→256→32) | None | **MISSING** |
| IoU prediction | `iou_prediction_head` 3-layer MLP (256→256→4) | `torch.ones` constant | **WRONG** |
| Multi-scale mask conv | `conv_s0` (32ch), `conv_s1` (64ch) Conv2d(k=1) | None | **MISSING** |
| Object score | `obj_score_token` + `pred_obj_score_head` MLP | None | **MISSING** |
| Shared image embedding | `shared_image_embedding.positional_embedding` [2,128] | None | **MISSING** |

## Honest Status Matrix

| Gate | Current Status | Evidence |
|------|---------------|----------|
| Official HF architecture faithfully implemented | **FAIL — 0 of 12 major components match** | Architecture-difference table above |
| Real pretrained weights | **FAIL — all torch.randn** | No HF checkpoint loaded |
| Patch embedding | **FAIL — wrong op** | linear vs Conv2d(k7,s4,p3) |
| Hiera blocks | **FAIL — simplified 4x SDPA vs 12 real blocks** | Missing windowing, layernorm, MLP, residuals |
| Native stage transitions | **FAIL — avg_pool2d host fallback** | Missing query pooling or strided conv |
| Image neck/FPN | **FAIL — missing entirely** | No Sam2VisionNeck |
| Point prompts | **FAIL — wrong encoding** | ttnn.linear vs positional+embed lookups |
| Box prompts | **FAIL — API only, no implementation** | Accepts parameter but does nothing |
| Mask prompts | **FAIL — missing entirely** | No dense mask pathway |
| Complete mask decoder | **FAIL — simplified cross-attn vs 2-block transformer** | Missing tokens, upscaling, hypernetworks, heads |
| Real sample image | **FAIL — never tested** | No image pipeline |
| Entire model on device | **FAIL — avg_pool2d + torch.zeros fallbacks** | Host transfers during forward |
| N150/N300 run | **UNTESTED** | No hardware available, not in CI |
| Stage 2 sharding | **FAIL — L1_MEMORY_CONFIG only, no ShardSpec** | No per-tensor sharding decisions |
| Performance report | **FAIL — missing** | No test_performance.py |

## Conclusion

Our implementation is a **simplified toy prototype** that shares only the channel dimensions [96,192,384,768] and the rough pipeline structure (encode→prompt→decode) with the real SAM2. **Every single component needs to be rewritten** to match the official HF architecture. There is nothing to optimize here — the entire foundation must be rebuilt.
