# BRINGUP LOG: rednote-hilab/dots.ocr

**Model:** `rednote-hilab/dots.ocr`
**Slug:** `rednote_hilab_dots.ocr`
**Target Device:** p150 (blackhole)
**Started:** 2026-05-29T00:11:46Z
**Updated:** 2026-05-29T00:30:23Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| vision_patch_embed | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_patch_embed | ttnn | pending | — | 0 |  |
| vision_patch_embed | debug | n/a | — | 0 |  |
| vision_patch_embed | optimization | pending | — | 0 |  |
| vision_patch_embed | real_weights | pending | — | 0 |  |
| vision_rmsnorm | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_rmsnorm | ttnn | pending | — | 0 |  |
| vision_rmsnorm | debug | n/a | — | 0 |  |
| vision_rmsnorm | optimization | pending | — | 0 |  |
| vision_rmsnorm | real_weights | pending | — | 0 |  |
| vision_attention | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_attention | ttnn | pending | — | 0 |  |
| vision_attention | debug | n/a | — | 0 |  |
| vision_attention | optimization | pending | — | 0 |  |
| vision_attention | real_weights | pending | — | 0 |  |
| vision_mlp | reference | done | 1.000000 | 0 | reference vs HF (eager) module, PCC=1.0; golden saved |
| vision_mlp | ttnn | pending | — | 0 |  |
| vision_mlp | debug | n/a | — | 0 |  |
| vision_mlp | optimization | pending | — | 0 |  |
| vision_mlp | real_weights | pending | — | 0 |  |
| vision_block | reference | done | 1.000000 | 0 | DotsVisionBlock: pre-norm RMSNorm -> fused-QKV 2D-RoPE bidirectional attn -> residual -> RMSNorm -> SwiGLU -> residual. PCC=1.0 vs HF eager. |
| vision_block | ttnn | pending | — | 0 |  |
| vision_block | debug | n/a | — | 0 |  |
| vision_block | optimization | pending | — | 0 |  |
| vision_block | real_weights | pending | — | 0 |  |
| vision_patch_merger | reference | done | 1.000000 | 0 | PatchMerger: LayerNorm(eps=1e-6,bias) -> view(merge**2) -> Linear -> GELU -> Linear (all biased). PCC=1.0 vs HF. |
| vision_patch_merger | ttnn | pending | — | 0 |  |
| vision_patch_merger | debug | n/a | — | 0 |  |
| vision_patch_merger | optimization | pending | — | 0 |  |
| vision_patch_merger | real_weights | pending | — | 0 |  |
| vision_tower | reference | done | 1.000000 | 0 | Full DotsVisionTransformer tested at REDUCED 2 layers (full=42, grid 1x4x4=16 patches, bf16=False fp32 path). PCC=1.0 vs HF. Full-depth check deferred to real_weights. |
| vision_tower | ttnn | pending | — | 0 |  |
| vision_tower | debug | n/a | — | 0 |  |
| vision_tower | optimization | pending | — | 0 |  |
| vision_tower | real_weights | pending | — | 0 |  |
| embedding | reference | done | 1.000000 | 0 | Qwen2 token embedding (vocab 151936, hidden 1536) lookup. PCC=1.0 vs nn.Embedding. |
| embedding | ttnn | pending | — | 0 |  |
| embedding | debug | n/a | — | 0 |  |
| embedding | optimization | pending | — | 0 |  |
| embedding | real_weights | pending | — | 0 |  |
| rmsnorm | reference | pending | — | 0 |  |
| rmsnorm | ttnn | pending | — | 0 |  |
| rmsnorm | debug | n/a | — | 0 |  |
| rmsnorm | optimization | pending | — | 0 |  |
| rmsnorm | real_weights | pending | — | 0 |  |
| rope | reference | pending | — | 0 |  |
| rope | ttnn | pending | — | 0 |  |
| rope | debug | n/a | — | 0 |  |
| rope | optimization | pending | — | 0 |  |
| rope | real_weights | pending | — | 0 |  |
| attention | reference | pending | — | 0 |  |
| attention | ttnn | pending | — | 0 |  |
| attention | debug | n/a | — | 0 |  |
| attention | optimization | pending | — | 0 |  |
| attention | real_weights | pending | — | 0 |  |
| mlp | reference | pending | — | 0 |  |
| mlp | ttnn | pending | — | 0 |  |
| mlp | debug | n/a | — | 0 |  |
| mlp | optimization | pending | — | 0 |  |
| mlp | real_weights | pending | — | 0 |  |
| decoder_layer | reference | pending | — | 0 |  |
| decoder_layer | ttnn | pending | — | 0 |  |
| decoder_layer | debug | n/a | — | 0 |  |
| decoder_layer | optimization | pending | — | 0 |  |
| decoder_layer | real_weights | pending | — | 0 |  |
| lm_head | reference | pending | — | 0 |  |
| lm_head | ttnn | pending | — | 0 |  |
| lm_head | debug | n/a | — | 0 |  |
| lm_head | optimization | pending | — | 0 |  |
| lm_head | real_weights | pending | — | 0 |  |
| language_model | reference | pending | — | 0 |  |
| language_model | ttnn | pending | — | 0 |  |
| language_model | debug | n/a | — | 0 |  |
| language_model | optimization | pending | — | 0 |  |
| language_model | real_weights | pending | — | 0 |  |

## Use cases

| Name | Input | Output | needs_ar | Generation | Perf |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ocr | image | text | yes | pending | pending |

## Recent Ticks

- tick 1 (2026-05-29T00:15:04Z): architecture[all] — ok
- tick 2 (2026-05-29T00:21:12Z): reference[vision_patch_embed,vision_rmsnorm,vision_attention,vision_mlp] — ok
- tick 3 (2026-05-29T00:30:23Z): reference[vision_block,vision_patch_merger,vision_tower,embedding] — ok

## Host-Resident Exceptions

- **vision_patch_embed**: DotsPatchEmbed is a single Conv2d(3,1536,k=14,s=14) over patchified pixels followed by RMSNorm; Conv2d patchify is a one-shot host-side im2col+matmul in the Qwen-VL TTNN demos (qwen25_vl runs patch embed on host then moves tokens to device). Cheap relative to the 42-layer trunk and runs once per image. (ref: models/demos/qwen25_vl/tt/model.py)
