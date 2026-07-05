# Bring-up plan: `coqui/XTTS-v2`

Backend template: **XTTS-v2 (multilingual TTS)** at `models/demos/xtts_v2` (canonical HF id: `/local/ttuser/apande/models/XTTS-v2-hf`).

**Summary:** 3 REUSE · 29 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `/local/ttuser/apande/models/XTTS-v2-hf` and re-run for a sharper diff.

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **NEW** | `g_p_t` | `—` | `—` |
| **NEW** | `g_p_t2_inference_model` | `—` | `—` |
| **NEW** | `g_p_t2_model` | `—` | `—` |
| **NEW** | `g_p_t2_block` | `—` | `—` |
| **NEW** | `hifi_decoder` | `—` | `—` |
| **NEW** | `res_net_speaker_encoder` | `—` | `—` |
| **NEW** | `s_e_basic_block` | `—` | `—` |
| **NEW** | `conv1_d` | `—` | `—` |
| **REUSE** | `g_p_t2_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `g_p_t2_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **NEW** | `hifigan_generator` | `—` | `—` |
| **NEW** | `s_e_layer` | `—` | `—` |
| **NEW** | `parametrization_list` | `—` | `—` |
| **NEW** | `weight_norm` | `—` | `—` |
| **NEW** | `parametrized_conv1d` | `—` | `—` |
| **NEW** | `res_block1` | `—` | `—` |
| **NEW** | `conditioning_encoder` | `—` | `—` |
| **NEW** | `attention_block` | `—` | `—` |
| **NEW** | `adaptive_avg_pool2d` | `—` | `—` |
| **NEW** | `perceiver_resampler` | `—` | `—` |
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `group_norm32` | `—` | `—` |
| **NEW** | `q_k_v_attention_legacy` | `—` | `—` |
| **NEW** | `parametrized_conv_transpose1d` | `—` | `—` |
| **NEW** | `attend` | `—` | `—` |
| **NEW** | `g_e_g_l_u` | `—` | `—` |
| **NEW** | `learned_position_embeddings` | `—` | `—` |
| **NEW** | `mel_spectrogram` | `—` | `—` |
| **NEW** | `dropout1d` | `—` | `—` |
| **NEW** | `instance_norm1d` | `—` | `—` |
| **NEW** | `mel_scale` | `—` | `—` |
| **NEW** | `pre_emphasis` | `—` | `—` |

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

### `g_p_t` — NEW
_module-tree: occ=1 leaves=660 sample_paths=['gpt']_

| field | new model | sibling |
|---|---|---|

### `g_p_t2_inference_model` — NEW
_module-tree: occ=1 leaves=308 sample_paths=['gpt.gpt_inference']_

| field | new model | sibling |
|---|---|---|

### `g_p_t2_model` — NEW
_module-tree: occ=1 leaves=303 sample_paths=['gpt.gpt']_

| field | new model | sibling |
|---|---|---|

### `g_p_t2_block` — NEW
_module-tree: occ=30 leaves=300 sample_paths=['gpt.gpt.h.0', 'gpt.gpt.h.1']_

| field | new model | sibling |
|---|---|---|

### `hifi_decoder` — NEW
_module-tree: occ=1 leaves=262 sample_paths=['hifigan_decoder']_

| field | new model | sibling |
|---|---|---|

### `res_net_speaker_encoder` — NEW
_module-tree: occ=1 leaves=179 sample_paths=['hifigan_decoder.speaker_encoder']_

| field | new model | sibling |
|---|---|---|

### `s_e_basic_block` — NEW
_module-tree: occ=16 leaves=166 sample_paths=['hifigan_decoder.speaker_encoder.layer1.0', 'hifigan_decoder.speaker_encoder.layer1.1']_

| field | new model | sibling |
|---|---|---|

### `conv1_d` — NEW
_module-tree: occ=120 leaves=120 sample_paths=['gpt.gpt.h.0.attn.c_attn', 'gpt.gpt.h.0.attn.c_proj']_

| field | new model | sibling |
|---|---|---|

### `g_p_t2_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=30 leaves=120 sample_paths=['gpt.gpt.h.0.attn', 'gpt.gpt.h.1.attn']_

| field | new model | sibling |
|---|---|---|

### `g_p_t2_m_l_p` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=30 leaves=120 sample_paths=['gpt.gpt.h.0.mlp', 'gpt.gpt.h.1.mlp']_

| field | new model | sibling |
|---|---|---|

### `hifigan_generator` — NEW
_module-tree: occ=1 leaves=83 sample_paths=['hifigan_decoder.waveform_decoder']_

| field | new model | sibling |
|---|---|---|

### `s_e_layer` — NEW
_module-tree: occ=16 leaves=80 sample_paths=['hifigan_decoder.speaker_encoder.layer1.0.se', 'hifigan_decoder.speaker_encoder.layer1.1.se']_

| field | new model | sibling |
|---|---|---|

### `parametrization_list` — NEW
_module-tree: occ=76 leaves=76 sample_paths=['hifigan_decoder.waveform_decoder.ups.0.parametrizations.weight', 'hifigan_decoder.waveform_decoder.ups.1.parametrizations.weight']_

| field | new model | sibling |
|---|---|---|

### `weight_norm` — NEW
_module-tree: occ=76 leaves=76 sample_paths=['hifigan_decoder.waveform_decoder.ups.0.parametrizations.weight.0', 'hifigan_decoder.waveform_decoder.ups.1.parametrizations.weight.0']_

| field | new model | sibling |
|---|---|---|

### `parametrized_conv1d` — NEW
_module-tree: occ=72 leaves=72 sample_paths=['hifigan_decoder.waveform_decoder.resblocks.0.convs1.0', 'hifigan_decoder.waveform_decoder.resblocks.0.convs1.1']_

| field | new model | sibling |
|---|---|---|

### `res_block1` — NEW
_module-tree: occ=12 leaves=72 sample_paths=['hifigan_decoder.waveform_decoder.resblocks.0', 'hifigan_decoder.waveform_decoder.resblocks.1']_

| field | new model | sibling |
|---|---|---|

### `conditioning_encoder` — NEW
_module-tree: occ=1 leaves=25 sample_paths=['gpt.conditioning_encoder']_

| field | new model | sibling |
|---|---|---|

### `attention_block` — NEW
_module-tree: occ=6 leaves=24 sample_paths=['gpt.conditioning_encoder.attn.0', 'gpt.conditioning_encoder.attn.1']_

| field | new model | sibling |
|---|---|---|

### `adaptive_avg_pool2d` — NEW
_module-tree: occ=16 leaves=16 sample_paths=['hifigan_decoder.speaker_encoder.layer1.0.se.avg_pool', 'hifigan_decoder.speaker_encoder.layer1.1.se.avg_pool']_

| field | new model | sibling |
|---|---|---|

### `perceiver_resampler` — NEW
_module-tree: occ=1 leaves=16 sample_paths=['gpt.conditioning_perceiver']_

| field | new model | sibling |
|---|---|---|

### `attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=2 leaves=8 sample_paths=['gpt.conditioning_perceiver.layers.0.0', 'gpt.conditioning_perceiver.layers.1.0']_

| field | new model | sibling |
|---|---|---|

### `group_norm32` — NEW
_module-tree: occ=6 leaves=6 sample_paths=['gpt.conditioning_encoder.attn.0.norm', 'gpt.conditioning_encoder.attn.1.norm']_

| field | new model | sibling |
|---|---|---|

### `q_k_v_attention_legacy` — NEW
_module-tree: occ=6 leaves=6 sample_paths=['gpt.conditioning_encoder.attn.0.attention', 'gpt.conditioning_encoder.attn.1.attention']_

| field | new model | sibling |
|---|---|---|

### `parametrized_conv_transpose1d` — NEW
_module-tree: occ=4 leaves=4 sample_paths=['hifigan_decoder.waveform_decoder.ups.0', 'hifigan_decoder.waveform_decoder.ups.1']_

| field | new model | sibling |
|---|---|---|

### `attend` — NEW
_module-tree: occ=2 leaves=2 sample_paths=['gpt.conditioning_perceiver.layers.0.0.attend', 'gpt.conditioning_perceiver.layers.1.0.attend']_

| field | new model | sibling |
|---|---|---|

### `g_e_g_l_u` — NEW
_module-tree: occ=2 leaves=2 sample_paths=['gpt.conditioning_perceiver.layers.0.1.1', 'gpt.conditioning_perceiver.layers.1.1.1']_

| field | new model | sibling |
|---|---|---|

### `learned_position_embeddings` — NEW
_module-tree: occ=2 leaves=2 sample_paths=['gpt.mel_pos_embedding', 'gpt.text_pos_embedding']_

| field | new model | sibling |
|---|---|---|

### `mel_spectrogram` — NEW
_module-tree: occ=1 leaves=2 sample_paths=['hifigan_decoder.speaker_encoder.torch_spec.1']_

| field | new model | sibling |
|---|---|---|

### `dropout1d` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['gpt.conditioning_dropout']_

| field | new model | sibling |
|---|---|---|

### `instance_norm1d` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['hifigan_decoder.speaker_encoder.instancenorm']_

| field | new model | sibling |
|---|---|---|

### `mel_scale` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['hifigan_decoder.speaker_encoder.torch_spec.1.mel_scale']_

| field | new model | sibling |
|---|---|---|

### `pre_emphasis` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['hifigan_decoder.speaker_encoder.torch_spec.0']_

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
