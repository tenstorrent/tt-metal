# Bring-up plan: `meituan-longcat/LongCat-Image`

Backend template: **Stable Diffusion 1.4** at `models/demos/vision/generative/stable_diffusion` (canonical HF id: `CompVis/stable-diffusion-v1-4`).

**Summary:** 9 REUSE · 25 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `CompVis/stable-diffusion-v1-4` and re-run for a sharper diff.

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **REUSE** | `self_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mlp` | `models/tt_transformers/tt/mlp.py` | `—` |
| **NEW** | `qwen2_v_l_for_conditional_generation` | `—` | `—` |
| **NEW** | `qwen2_v_l_model` | `—` | `—` |
| **NEW** | `long_cat_image_transformer2_d_model` | `—` | `—` |
| **NEW** | `qwen2_v_l_text_model` | `—` | `—` |
| **NEW** | `qwen2_v_l_decoder_layer` | `—` | `—` |
| **NEW** | `long_cat_image_transformer_block` | `—` | `—` |
| **NEW** | `qwen2_vision_transformer_pretrained_model` | `—` | `—` |
| **NEW** | `qwen2_v_l_vision_block` | `—` | `—` |
| **REUSE** | `long_cat_image_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `long_cat_image_single_transformer_block` | `—` | `—` |
| **NEW** | `autoencoder_k_l` | `—` | `—` |
| **NEW** | `resnet_block2_d` | `—` | `—` |
| **REUSE** | `qwen2_v_l_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `qwen2_v_l_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `qwen2_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `qwen2_v_l_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `decoder` | `—` | `—` |
| **NEW** | `up_decoder_block2_d` | `—` | `—` |
| **NEW** | `encoder` | `—` | `—` |
| **REUSE** | `qwen2_v_l_vision_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `ada_layer_norm_zero` | `—` | `—` |
| **NEW** | `ada_layer_norm_zero_single` | `—` | `—` |
| **NEW** | `feed_forward` | `—` | `—` |
| **NEW** | `down_encoder_block2_d` | `—` | `—` |
| **NEW** | `u_net_mid_block2_d` | `—` | `—` |
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `long_cat_image_timestep_embeddings` | `—` | `—` |
| **NEW** | `qwen2_v_l_patch_merger` | `—` | `—` |
| **NEW** | `ada_layer_norm_continuous` | `—` | `—` |
| **NEW** | `downsample2_d` | `—` | `—` |
| **NEW** | `timestep_embedding` | `—` | `—` |
| **NEW** | `upsample2_d` | `—` | `—` |

## Shared modules (always reusable, no copy needed)

| Purpose | tt-metal path |
|---|---|
| LayerNorm / RMSNorm | `models/common/rmsnorm.py` |
| LightweightModule base | `models/common/lightweightmodule.py` |
| Tensor helpers | `models/common/tensor_utils.py` |
| Generic utility funcs | `models/common/utility_functions.py` |

## Action by status

- **REUSE**: import / call the sibling's tt-module unchanged. Weight names match. The global PCC gate enforces this — if it fails, `force_adapt_all` demotes the REUSE component to NEW and the brain iterates per-component.
- **NEW**: write/adapt the TTNN port. A stub file is generated under `_stubs/` (torch fallback by default), then progressively rewritten to native ttnn through per-component PCC iteration. If a sibling tt-file with the same role exists, the agent reuses its layout and updates shape constants (hidden_size, num_heads, intermediate_size, eps); otherwise it writes from scratch against the HF reference.

## Per-component shape diff

### `self_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0._

| field | new model | sibling |
|---|---|---|

### `mlp` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh._

| field | new model | sibling |
|---|---|---|

### `qwen2_v_l_for_conditional_generation` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=546 sample_paths=['text_encoder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_v_l_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=545 sample_paths=['text_encoder.model'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `long_cat_image_transformer2_d_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=501 sample_paths=['transformer'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_v_l_text_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=283 sample_paths=['text_encoder.model.language_model'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_v_l_decoder_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=28 leaves=280 sample_paths=['text_encoder.model.language_model.layers.0', 'text_encoder.model.language_model.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `long_cat_image_transformer_block` — NEW
_[supplemental module-tree pass] module-tree: occ=10 leaves=270 sample_paths=['transformer.transformer_blocks.0', 'transformer.transformer_blocks.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_vision_transformer_pretrained_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=262 sample_paths=['text_encoder.model.visual'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_v_l_vision_block` — NEW
_[supplemental module-tree pass] module-tree: occ=32 leaves=256 sample_paths=['text_encoder.model.visual.blocks.0', 'text_encoder.model.visual.blocks.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `long_cat_image_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=30 leaves=230 sample_paths=['transformer.transformer_blocks.0.attn', 'transformer.transformer_blocks.1.attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `long_cat_image_single_transformer_block` — NEW
_[supplemental module-tree pass] module-tree: occ=20 leaves=220 sample_paths=['transformer.single_transformer_blocks.0', 'transformer.single_transformer_blocks.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `autoencoder_k_l` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=174 sample_paths=['vae'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `resnet_block2_d` — NEW
_[supplemental module-tree pass] module-tree: occ=24 leaves=148 sample_paths=['vae.encoder.down_blocks.0.resnets.0', 'vae.encoder.down_blocks.0.resnets.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_v_l_m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=32 leaves=128 sample_paths=['text_encoder.model.visual.blocks.0.mlp', 'text_encoder.model.visual.blocks.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_v_l_r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=122 leaves=122 sample_paths=['text_encoder.model.visual.blocks.0.norm1', 'text_encoder.model.visual.blocks.0.norm2'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=28 leaves=112 sample_paths=['text_encoder.model.language_model.layers.0.mlp', 'text_encoder.model.language_model.layers.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_v_l_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=28 leaves=112 sample_paths=['text_encoder.model.language_model.layers.0.self_attn', 'text_encoder.model.language_model.layers.1.self_attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `decoder` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=99 sample_paths=['vae.decoder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `up_decoder_block2_d` — NEW
_[supplemental module-tree pass] module-tree: occ=4 leaves=77 sample_paths=['vae.decoder.up_blocks.0', 'vae.decoder.up_blocks.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `encoder` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=75 sample_paths=['vae.encoder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_v_l_vision_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=32 leaves=64 sample_paths=['text_encoder.model.visual.blocks.0.attn', 'text_encoder.model.visual.blocks.1.attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ada_layer_norm_zero` — NEW
_[supplemental module-tree pass] module-tree: occ=20 leaves=60 sample_paths=['transformer.transformer_blocks.0.norm1', 'transformer.transformer_blocks.0.norm1_context'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ada_layer_norm_zero_single` — NEW
_[supplemental module-tree pass] module-tree: occ=20 leaves=60 sample_paths=['transformer.single_transformer_blocks.0.norm', 'transformer.single_transformer_blocks.1.norm'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `feed_forward` — NEW
_[supplemental module-tree pass] module-tree: occ=20 leaves=60 sample_paths=['transformer.transformer_blocks.0.ff', 'transformer.transformer_blocks.0.ff_context'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `down_encoder_block2_d` — NEW
_[supplemental module-tree pass] module-tree: occ=4 leaves=53 sample_paths=['vae.encoder.down_blocks.0', 'vae.encoder.down_blocks.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `u_net_mid_block2_d` — NEW
_[supplemental module-tree pass] module-tree: occ=2 leaves=36 sample_paths=['vae.encoder.mid_block', 'vae.decoder.mid_block'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=2 leaves=12 sample_paths=['vae.encoder.mid_block.attentions.0', 'vae.decoder.mid_block.attentions.0'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `long_cat_image_timestep_embeddings` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=4 sample_paths=['transformer.time_embed'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen2_v_l_patch_merger` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=4 sample_paths=['text_encoder.model.visual.merger'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ada_layer_norm_continuous` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=3 sample_paths=['transformer.norm_out'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `downsample2_d` — NEW
_[supplemental module-tree pass] module-tree: occ=3 leaves=3 sample_paths=['vae.encoder.down_blocks.0.downsamplers.0', 'vae.encoder.down_blocks.1.downsamplers.0'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `timestep_embedding` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=3 sample_paths=['transformer.time_embed.timestep_embedder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `upsample2_d` — NEW
_[supplemental module-tree pass] module-tree: occ=3 leaves=3 sample_paths=['vae.decoder.up_blocks.0.upsamplers.0', 'vae.decoder.up_blocks.1.upsamplers.0'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
