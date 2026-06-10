# BRINGUP LOG: rednote-hilab/dots.ocr

**Model:** `rednote-hilab/dots.ocr`
**Slug:** `rednote_hilab_dots.ocr`
**Target Device:** qb (blackhole)
**Started:** 2026-06-10T00:12:02Z
**Updated:** 2026-06-10T02:43:31Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| vision_patch_embed | reference | done | 1.000000 | 0 | Conv2d 14x14 s14 3->1536 + RMSNorm, flat-patch input like HF |
| vision_patch_embed | ttnn | pending | — | 0 |  |
| vision_patch_embed | debug | n/a | — | 0 |  |
| vision_patch_embed | optimization | pending | — | 0 |  |
| vision_patch_embed | real_weights | pending | — | 0 |  |
| vision_rmsnorm | reference | done | 1.000000 | 0 | matches HF RMSNorm eps=1e-5, fp32 norm then scale |
| vision_rmsnorm | ttnn | pending | — | 0 |  |
| vision_rmsnorm | debug | n/a | — | 0 |  |
| vision_rmsnorm | optimization | pending | — | 0 |  |
| vision_rmsnorm | real_weights | pending | — | 0 |  |
| vision_attention | reference | done | 1.000000 | 0 | eager MHA 12h hd128 fused QKV no-bias, 2D rope, cu_seqlens mask |
| vision_attention | ttnn | pending | — | 0 |  |
| vision_attention | debug | n/a | — | 0 |  |
| vision_attention | optimization | pending | — | 0 |  |
| vision_attention | real_weights | pending | — | 0 |  |
| vision_mlp | reference | done | 1.000000 | 0 | SwiGLU fc1/fc3->fc2 no bias |
| vision_mlp | ttnn | pending | — | 0 |  |
| vision_mlp | debug | n/a | — | 0 |  |
| vision_mlp | optimization | pending | — | 0 |  |
| vision_mlp | real_weights | pending | — | 0 |  |
| vision_block | reference | done | 1.000000 | 0 | pre-norm residual block x+attn(norm1(x)); x+mlp(norm2(x)), real blocks.0 weights |
| vision_block | ttnn | pending | — | 0 |  |
| vision_block | debug | n/a | — | 0 |  |
| vision_block | optimization | pending | — | 0 |  |
| vision_block | real_weights | pending | — | 0 |  |
| patch_merger | reference | done | 1.000000 | 0 | LayerNorm(eps=1e-6) -> view(-1,6144) -> Linear -> GELU -> Linear 6144->1536, real merger weights |
| patch_merger | ttnn | pending | — | 0 |  |
| patch_merger | debug | n/a | — | 0 |  |
| patch_merger | optimization | pending | — | 0 |  |
| patch_merger | real_weights | pending | — | 0 |  |
| vision_transformer | reference | done | 1.000000 | 0 | full tower: patch_embed -> 42x block -> post_trunk_norm -> merger, all real fp32 weights, matches HF DotsVisionTransformer(bf16=False) |
| vision_transformer | ttnn | pending | — | 0 |  |
| vision_transformer | debug | n/a | — | 0 |  |
| vision_transformer | optimization | pending | — | 0 |  |
| vision_transformer | real_weights | pending | — | 0 |  |
| embedding | reference | done | 1.000000 | 0 | F.embedding lookup on model.embed_tokens.weight 151936x1536, untied |
| embedding | ttnn | pending | — | 0 |  |
| embedding | debug | n/a | — | 0 |  |
| embedding | optimization | pending | — | 0 |  |
| embedding | real_weights | pending | — | 0 |  |
| text_rmsnorm | reference | done | 1.000000 | 0 | Qwen2RMSNorm eps=1e-6, fp32 variance then weight*x, real layers.0 input_layernorm weight |
| text_rmsnorm | ttnn | pending | — | 0 |  |
| text_rmsnorm | debug | n/a | — | 0 |  |
| text_rmsnorm | optimization | pending | — | 0 |  |
| text_rmsnorm | real_weights | pending | — | 0 |  |
| text_attention | reference | done | 1.000000 | 0 | eager causal GQA 12Q/2KV hd128 qkv-bias o_proj-no-bias, rope theta=1e6, real layers.0 weights |
| text_attention | ttnn | pending | — | 0 |  |
| text_attention | debug | n/a | — | 0 |  |
| text_attention | optimization | pending | — | 0 |  |
| text_attention | real_weights | pending | — | 0 |  |
| text_mlp | reference | done | 1.000000 | 0 | Qwen2 SwiGLU 1536->8960->1536 no bias, real layers.0 weights |
| text_mlp | ttnn | pending | — | 0 |  |
| text_mlp | debug | n/a | — | 0 |  |
| text_mlp | optimization | pending | — | 0 |  |
| text_mlp | real_weights | pending | — | 0 |  |
| decoder_layer | reference | done | 1.000000 | 0 | Qwen2DecoderLayer pre-norm residual x+attn(ln1(x)); h+mlp(ln2(h)), real layers.0 weights |
| decoder_layer | ttnn | pending | — | 0 |  |
| decoder_layer | debug | n/a | — | 0 |  |
| decoder_layer | optimization | pending | — | 0 |  |
| decoder_layer | real_weights | pending | — | 0 |  |
| lm_head | reference | done | 1.000000 | 0 | untied Linear 1536->151936 no bias, real lm_head.weight, matches torch.nn.Linear |
| lm_head | ttnn | pending | — | 0 |  |
| lm_head | debug | n/a | — | 0 |  |
| lm_head | optimization | pending | — | 0 |  |
| lm_head | real_weights | pending | — | 0 |  |

## Use cases

| Name | Input | Output | needs_ar | Generation | Perf |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ocr | image | text | yes | pending | pending |

## Recent Ticks

- tick 1 (2026-06-10T02:15:14Z): architecture[all] — ok
- tick 2 (2026-06-10T02:23:55Z): reference[vision_patch_embed,vision_rmsnorm,vision_attention,vision_mlp] — ok
- tick 3 (2026-06-10T02:31:43Z): reference[vision_block,patch_merger,vision_transformer,embedding] — ok
- tick 4 (2026-06-10T02:38:20Z): reference[text_rmsnorm,text_attention,text_mlp,decoder_layer] — ok
- tick 5 (2026-06-10T02:43:31Z): reference[lm_head] — ok

## Host-Resident Exceptions

_None._
