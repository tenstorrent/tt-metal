# Bring-up plan: `microsoft/VibeVoice-1.5B`

Backend template: **XTTS-v2 (multilingual TTS)** at `models/demos/xtts_v2` (canonical HF id: `/local/ttuser/apande/models/XTTS-v2-hf`).
New `model_type` = `vibevoice`; sibling `model_type` = `None`.

**Summary:** 6 REUSE · 19 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `/local/ttuser/apande/models/XTTS-v2-hf` and re-run for a sharper diff.

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **NEW** | `block1_d` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `tokenizer_encoder` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `vibe_voice_acoustic_tokenizer_model` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `qwen2_model` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `qwen2_decoder_layer` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `f_f_n` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `tokenizer_decoder` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `vibe_voice_semantic_tokenizer_model` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `norm_conv1d` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `s_conv1d` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **REUSE** | `conv_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **NEW** | `convlayer` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **REUSE** | `qwen2_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `qwen2_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `qwen2_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **NEW** | `vibe_voice_diffusion_head` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `head_layer` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `feed_forward_network` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `norm_conv_transpose1d` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `s_conv_transpose1d` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `speech_connector` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `final_layer` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **NEW** | `timestep_embedder` | `—` | `transformers/src/transformers/models/vibevoice/modeling_vibevoice.py` |
| **REUSE** | `llama_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `qwen2_rotary_embedding` | `models/tt_transformers/tt/rope.py` | `—` |

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

### `block1_d` — NEW
_module-tree: occ=78 leaves=624 sample_paths=['acoustic_tokenizer.encoder.stages.0.0', 'acoustic_tokenizer.encoder.stages.0.1']_

| field | new model | sibling |
|---|---|---|

### `tokenizer_encoder` — NEW
_module-tree: occ=2 leaves=450 sample_paths=['acoustic_tokenizer.encoder', 'semantic_tokenizer.encoder']_

| field | new model | sibling |
|---|---|---|

### `vibe_voice_acoustic_tokenizer_model` — NEW
_module-tree: occ=1 leaves=450 sample_paths=['acoustic_tokenizer']_

| field | new model | sibling |
|---|---|---|

### `qwen2_model` — NEW
_module-tree: occ=1 leaves=283 sample_paths=['language_model']_

| field | new model | sibling |
|---|---|---|

### `qwen2_decoder_layer` — NEW
_module-tree: occ=28 leaves=280 sample_paths=['language_model.layers.0', 'language_model.layers.1']_

| field | new model | sibling |
|---|---|---|

### `f_f_n` — NEW
_module-tree: occ=78 leaves=234 sample_paths=['acoustic_tokenizer.encoder.stages.0.0.ffn', 'acoustic_tokenizer.encoder.stages.0.1.ffn']_

| field | new model | sibling |
|---|---|---|

### `tokenizer_decoder` — NEW
_module-tree: occ=1 leaves=225 sample_paths=['acoustic_tokenizer.decoder']_

| field | new model | sibling |
|---|---|---|

### `vibe_voice_semantic_tokenizer_model` — NEW
_module-tree: occ=1 leaves=225 sample_paths=['semantic_tokenizer']_

| field | new model | sibling |
|---|---|---|

### `norm_conv1d` — NEW
_module-tree: occ=96 leaves=192 sample_paths=['acoustic_tokenizer.encoder.downsample_layers.0.0.conv', 'acoustic_tokenizer.encoder.downsample_layers.1.0.conv']_

| field | new model | sibling |
|---|---|---|

### `s_conv1d` — NEW
_module-tree: occ=96 leaves=192 sample_paths=['acoustic_tokenizer.encoder.downsample_layers.0.0', 'acoustic_tokenizer.encoder.downsample_layers.1.0']_

| field | new model | sibling |
|---|---|---|

### `conv_r_m_s_norm` — REUSE
_reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=156 leaves=156 sample_paths=['acoustic_tokenizer.encoder.stages.0.0.norm', 'acoustic_tokenizer.encoder.stages.0.0.ffn_norm']_

| field | new model | sibling |
|---|---|---|

### `convlayer` — NEW
_module-tree: occ=78 leaves=156 sample_paths=['acoustic_tokenizer.encoder.stages.0.0.mixer', 'acoustic_tokenizer.encoder.stages.0.1.mixer']_

| field | new model | sibling |
|---|---|---|

### `qwen2_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=28 leaves=112 sample_paths=['language_model.layers.0.self_attn', 'language_model.layers.1.self_attn']_

| field | new model | sibling |
|---|---|---|

### `qwen2_m_l_p` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=28 leaves=112 sample_paths=['language_model.layers.0.mlp', 'language_model.layers.1.mlp']_

| field | new model | sibling |
|---|---|---|

### `qwen2_r_m_s_norm` — REUSE
_reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=57 leaves=57 sample_paths=['language_model.layers.0.input_layernorm', 'language_model.layers.0.post_attention_layernorm']_

| field | new model | sibling |
|---|---|---|

### `vibe_voice_diffusion_head` — NEW
_module-tree: occ=1 leaves=37 sample_paths=['prediction_head']_

| field | new model | sibling |
|---|---|---|

### `head_layer` — NEW
_module-tree: occ=4 leaves=28 sample_paths=['prediction_head.layers.0', 'prediction_head.layers.1']_

| field | new model | sibling |
|---|---|---|

### `feed_forward_network` — NEW
_module-tree: occ=4 leaves=16 sample_paths=['prediction_head.layers.0.ffn', 'prediction_head.layers.1.ffn']_

| field | new model | sibling |
|---|---|---|

### `norm_conv_transpose1d` — NEW
_module-tree: occ=6 leaves=12 sample_paths=['acoustic_tokenizer.decoder.upsample_layers.1.0.convtr', 'acoustic_tokenizer.decoder.upsample_layers.2.0.convtr']_

| field | new model | sibling |
|---|---|---|

### `s_conv_transpose1d` — NEW
_module-tree: occ=6 leaves=12 sample_paths=['acoustic_tokenizer.decoder.upsample_layers.1.0', 'acoustic_tokenizer.decoder.upsample_layers.2.0']_

| field | new model | sibling |
|---|---|---|

### `speech_connector` — NEW
_module-tree: occ=2 leaves=6 sample_paths=['acoustic_connector', 'semantic_connector']_

| field | new model | sibling |
|---|---|---|

### `final_layer` — NEW
_module-tree: occ=1 leaves=4 sample_paths=['prediction_head.final_layer']_

| field | new model | sibling |
|---|---|---|

### `timestep_embedder` — NEW
_module-tree: occ=1 leaves=3 sample_paths=['prediction_head.t_embedder']_

| field | new model | sibling |
|---|---|---|

### `llama_r_m_s_norm` — REUSE
_reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=2 leaves=2 sample_paths=['acoustic_connector.norm', 'semantic_connector.norm']_

| field | new model | sibling |
|---|---|---|

### `qwen2_rotary_embedding` — REUSE
_reuse_registry: standard_rope -> models/tt_transformers/tt/rope.py::RotaryEmbedding (REUSE). derived from compatibility.py BUILDING_BLOCKS 'Standard RoPE'. | module-tree: occ=1 leaves=1 sample_paths=['language_model.rotary_emb']_

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
