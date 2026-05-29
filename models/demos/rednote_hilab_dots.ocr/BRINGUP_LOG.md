# BRINGUP LOG: rednote-hilab/dots.ocr

**Model:** `rednote-hilab/dots.ocr`
**Slug:** `rednote_hilab_dots.ocr`
**Target Device:** p150 (blackhole)
**Started:** 2026-05-29T00:11:46Z
**Updated:** 2026-05-29T01:27:01Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| vision_patch_embed | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_patch_embed | ttnn | pending | — | 0 |  |
| vision_patch_embed | debug | n/a | — | 0 |  |
| vision_patch_embed | optimization | pending | — | 0 |  |
| vision_patch_embed | real_weights | pending | — | 0 |  |
| vision_rmsnorm | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_rmsnorm | ttnn | done | 0.999995 | 0 | ttnn.rms_norm eps=1e-5 HiFi4+fp32_dest_acc bf16 DRAM TILE; PCC 0.99999 vs seed-0 golden on p150 |
| vision_rmsnorm | debug | n/a | — | 0 |  |
| vision_rmsnorm | optimization | pending | — | 0 |  |
| vision_rmsnorm | real_weights | pending | — | 0 |  |
| vision_attention | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_attention | ttnn | done | 0.999988 | 0 | manual SDPA chain (qkv linear -> nlp_create_qkv_heads -> 2D RoPE -> matmul+softmax+matmul w/ block-diagonal additive mask -> nlp_concat_heads -> proj). HiFi4+fp32_dest_acc, bf16. PCC=0.999988 vs golden. Guard ok. |
| vision_attention | debug | n/a | — | 0 |  |
| vision_attention | optimization | pending | — | 0 |  |
| vision_attention | real_weights | pending | — | 0 |  |
| vision_mlp | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_mlp | ttnn | done | 0.999986 | 0 | fused gate(fc1)/up(fc3) ttnn.linear -> ttnn.silu(gate)*up -> down(fc2) ttnn.linear. No bias. HiFi4+fp32_dest_acc bf16 DRAM TILE. PCC=0.9999855 vs golden. Guard ok. |
| vision_mlp | debug | n/a | — | 0 |  |
| vision_mlp | optimization | pending | — | 0 |  |
| vision_mlp | real_weights | pending | — | 0 |  |
| vision_block | reference | done | 1.000000 | 0 | DotsVisionBlock: pre-norm RMSNorm -> fused-QKV 2D-RoPE bidirectional attn -> residual -> RMSNorm -> SwiGLU -> residual. PCC=1.0 vs HF eager. |
| vision_block | ttnn | done | 0.999996 | 0 | First composite: pre-norm residual h=h+attn(norm1(h)); h=h+mlp(norm2(h)). Reuses TtVisionRMSNorm/Attention/MLP leaves by file-path import. HiFi4+fp32_dest_acc, bf16. Guard ok. |
| vision_block | debug | n/a | — | 0 |  |
| vision_block | optimization | pending | — | 0 |  |
| vision_block | real_weights | pending | — | 0 |  |
| vision_patch_merger | reference | done | 1.000000 | 0 | PatchMerger: LayerNorm(eps=1e-6,bias) -> view(merge**2) -> Linear -> GELU -> Linear (all biased). PCC=1.0 vs HF. |
| vision_patch_merger | ttnn | done | 0.999990 | 0 | PatchMerger: ttnn.layer_norm (weight+bias, eps=1e-6) -> ttnn.reshape group 4 patches (1536->6144) -> ttnn.linear+bias -> ttnn.gelu -> ttnn.linear+bias. LN gamma/beta laid out [1,1,dim//32,32] row-major. HiFi4+fp32_dest_acc bf16 DRAM TILE. PCC=0.99999 vs golden. Guard ok. |
| vision_patch_merger | debug | n/a | — | 0 |  |
| vision_patch_merger | optimization | pending | — | 0 |  |
| vision_patch_merger | real_weights | pending | — | 0 |  |
| vision_tower | reference | done | 1.000000 | 0 | Full DotsVisionTransformer tested at REDUCED 2 layers (full=42, grid 1x4x4=16 patches, bf16=False fp32 path). PCC=1.0 vs HF. Full-depth check deferred to real_weights. |
| vision_tower | ttnn | pending | — | 0 |  |
| vision_tower | debug | n/a | — | 0 |  |
| vision_tower | optimization | pending | — | 0 |  |
| vision_tower | real_weights | pending | — | 0 |  |
| embedding | reference | done | 1.000000 | 0 | Qwen2 token embedding (vocab 151936, hidden 1536) lookup. PCC=1.0 vs nn.Embedding. |
| embedding | ttnn | done | 0.999999 | 0 | ttnn.embedding gather; weight [151936,1536] bf16 ROW_MAJOR in DRAM; uint32 row-major ids -> TILE output. HiFi4+fp32_dest_acc preset (no matmul; exact gather). PCC=0.9999986 vs golden on p150. Guard ok. |
| embedding | debug | n/a | — | 0 |  |
| embedding | optimization | pending | — | 0 |  |
| embedding | real_weights | pending | — | 0 |  |
| rmsnorm | reference | done | 1.000000 | 0 | Qwen2RMSNorm (eps 1e-6): fp32 normalize -> cast -> weight*x. PCC=1.0 vs HF Qwen2RMSNorm. |
| rmsnorm | ttnn | done | 0.999995 | 0 | Qwen2 LM RMSNorm eps=1e-6 (vs vision 1e-5). ttnn.rms_norm HiFi4+fp32_dest_acc bf16 DRAM TILE; weight reshaped [1,1,dim//32,32] row-major. Mirror of TtVisionRMSNorm. PCC=0.99999 vs golden on p150. Guard ok. |
| rmsnorm | debug | n/a | — | 0 |  |
| rmsnorm | optimization | pending | — | 0 |  |
| rmsnorm | real_weights | pending | — | 0 |  |
| rope | reference | done | 1.000000 | 0 | Qwen2RotaryEmbedding (theta 1e6, head_dim 128, default rope): position_ids -> (cos,sin). PCC cos=1.0 sin=1.0 vs HF. |
| rope | ttnn | done | 1.000000 | 0 | Qwen2 LM RoPE cos/sin tables (theta 1e6, head_dim 128). On-device outer(pos,inv_freq)->cat->cos/sin, fp32. PCC cos=1.0 sin=1.0 vs golden. |
| rope | debug | n/a | — | 0 |  |
| rope | optimization | pending | — | 0 |  |
| rope | real_weights | pending | — | 0 |  |
| attention | reference | done | 1.000000 | 0 | Qwen2Attention (GQA 12/2, head_dim 128, QKV bias, o_proj no bias, 1D RoPE, causal eager). PCC=1.0 vs HF eager. |
| attention | ttnn | done | 0.999989 | 0 | GQA 12/2 head_dim 128, QKV bias, o_proj no bias, 1D RoPE theta 1e6, causal additive mask. Manual head split (fused cat[Wq,Wk,Wv]), repeat_kv n_rep=6, HiFi4+fp32_dest_acc. |
| attention | debug | n/a | — | 0 |  |
| attention | optimization | pending | — | 0 |  |
| attention | real_weights | pending | — | 0 |  |
| mlp | reference | done | 1.000000 | 0 | Qwen2MLP SwiGLU SiLU (gate/up/down 1536<->8960, no bias). PCC=1.0 vs HF. |
| mlp | ttnn | pending | — | 0 |  |
| mlp | debug | n/a | — | 0 |  |
| mlp | optimization | pending | — | 0 |  |
| mlp | real_weights | pending | — | 0 |  |
| decoder_layer | reference | done | 1.000000 | 0 | Qwen2DecoderLayer: input RMSNorm -> GQA self-attn (12q/2kv, head_dim128, RoPE theta 1e6, QKV bias) -> residual -> post-attn RMSNorm -> SwiGLU MLP -> residual. PCC=1.0 vs HF Qwen2DecoderLayer. |
| decoder_layer | ttnn | pending | — | 0 |  |
| decoder_layer | debug | n/a | — | 0 |  |
| decoder_layer | optimization | pending | — | 0 |  |
| decoder_layer | real_weights | pending | — | 0 |  |
| lm_head | reference | done | 1.000000 | 0 | Untied Linear hidden 1536 -> vocab 151936, no bias. PCC=1.0 vs nn.Linear. |
| lm_head | ttnn | pending | — | 0 |  |
| lm_head | debug | n/a | — | 0 |  |
| lm_head | optimization | pending | — | 0 |  |
| lm_head | real_weights | pending | — | 0 |  |
| language_model | reference | done | 1.000000 | 0 | Full Qwen2ForCausalLM tested at REDUCED 2 layers (full=28), seq 64: embed -> 2x decoder_layer -> final RMSNorm -> lm_head. PCC=1.0 vs HF. Full 28-layer check deferred to real_weights. |
| language_model | ttnn | pending | — | 0 |  |
| language_model | debug | n/a | — | 0 |  |
| language_model | optimization | pending | — | 0 |  |
| language_model | real_weights | pending | — | 0 |  |

## Use cases

| Name | Input | Output | needs_ar | Generation | Perf |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ocr | image | text | yes | pending | pending |

## Recent Ticks

- tick 8 (2026-05-29T00:52:15Z): device[vision_attention] — ok
- tick 9 (2026-05-29T00:57:01Z): device[vision_mlp] — ok
- tick 9 (2026-05-29T00:58:10Z): device[vision_mlp] — ok
- tick 10 (2026-05-29T01:03:33Z): device[vision_block] — ok
- tick 10 (2026-05-29T01:04:12Z): device[vision_block] — ok
- tick 11 (2026-05-29T01:10:25Z): device[vision_patch_merger] — ok
- tick 12 (2026-05-29T01:13:48Z): device[embedding] — ok
- tick 13 (2026-05-29T01:17:43Z): device[rmsnorm] — ok
- tick 14 (2026-05-29T01:21:18Z): device[rope] — ok
- tick 15 (2026-05-29T01:27:01Z): device[attention] — ok

## Host-Resident Exceptions

- **vision_patch_embed**: DotsPatchEmbed is a single Conv2d(3,1536,k=14,s=14) over patchified pixels followed by RMSNorm; Conv2d patchify is a one-shot host-side im2col+matmul in the Qwen-VL TTNN demos (qwen25_vl runs patch embed on host then moves tokens to device). Cheap relative to the 42-layer trunk and runs once per image. (ref: models/demos/qwen25_vl/tt/model.py)
