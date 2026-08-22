# Bring-up plan: `hexgrad/Kokoro-82M`

Backend template: **XTTS-v2 (multilingual TTS)** at `models/demos/xtts_v2` (canonical HF id: `/local/ttuser/apande/models/XTTS-v2-hf`).

**Summary:** 1 REUSE · 24 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `/local/ttuser/apande/models/XTTS-v2-hf` and re-run for a sharper diff.

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **NEW** | `decoder` | `—` | `—` |
| **NEW** | `generator` | `—` | `—` |
| **NEW** | `ada_i_n_res_block1` | `—` | `—` |
| **NEW** | `ada_i_n1d` | `—` | `—` |
| **NEW** | `adain_res_blk1d` | `—` | `—` |
| **NEW** | `prosody_predictor` | `—` | `—` |
| **NEW** | `instance_norm1d` | `—` | `—` |
| **NEW** | `custom_albert` | `—` | `—` |
| **NEW** | `text_encoder` | `—` | `—` |
| **NEW** | `albert_transformer` | `—` | `—` |
| **NEW** | `albert_layer` | `—` | `—` |
| **NEW** | `albert_layer_group` | `—` | `—` |
| **NEW** | `up_sample1d` | `—` | `—` |
| **REUSE** | `albert_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `duration_encoder` | `—` | `—` |
| **NEW** | `l_s_t_m` | `—` | `—` |
| **NEW** | `albert_embeddings` | `—` | `—` |
| **NEW** | `ada_layer_norm` | `—` | `—` |
| **NEW** | `source_module_hn_n_s_f` | `—` | `—` |
| **NEW** | `leaky_re_l_u` | `—` | `—` |
| **NEW** | `custom_s_t_f_t` | `—` | `—` |
| **NEW** | `linear_norm` | `—` | `—` |
| **NEW** | `reflection_pad1d` | `—` | `—` |
| **NEW** | `sine_gen` | `—` | `—` |
| **NEW** | `upsample` | `—` | `—` |

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

### `decoder` — NEW
_module-tree: occ=1 leaves=229 sample_paths=['decoder']_

| field | new model | sibling |
|---|---|---|

### `generator` — NEW
_module-tree: occ=1 leaves=171 sample_paths=['decoder.generator']_

| field | new model | sibling |
|---|---|---|

### `ada_i_n_res_block1` — NEW
_module-tree: occ=8 leaves=160 sample_paths=['decoder.generator.noise_res.0', 'decoder.generator.noise_res.1']_

| field | new model | sibling |
|---|---|---|

### `ada_i_n1d` — NEW
_module-tree: occ=70 leaves=140 sample_paths=['predictor.F0.0.norm1', 'predictor.F0.0.norm2']_

| field | new model | sibling |
|---|---|---|

### `adain_res_blk1d` — NEW
_module-tree: occ=11 leaves=117 sample_paths=['predictor.F0.0', 'predictor.F0.1']_

| field | new model | sibling |
|---|---|---|

### `prosody_predictor` — NEW
_module-tree: occ=1 leaves=73 sample_paths=['predictor']_

| field | new model | sibling |
|---|---|---|

### `instance_norm1d` — NEW
_module-tree: occ=70 leaves=70 sample_paths=['predictor.F0.0.norm1.norm', 'predictor.F0.0.norm2.norm']_

| field | new model | sibling |
|---|---|---|

### `custom_albert` — NEW
_module-tree: occ=1 leaves=20 sample_paths=['bert']_

| field | new model | sibling |
|---|---|---|

### `text_encoder` — NEW
_module-tree: occ=1 leaves=14 sample_paths=['text_encoder']_

| field | new model | sibling |
|---|---|---|

### `albert_transformer` — NEW
_module-tree: occ=1 leaves=13 sample_paths=['bert.encoder']_

| field | new model | sibling |
|---|---|---|

### `albert_layer` — NEW
_module-tree: occ=1 leaves=12 sample_paths=['bert.encoder.albert_layer_groups.0.albert_layers.0']_

| field | new model | sibling |
|---|---|---|

### `albert_layer_group` — NEW
_module-tree: occ=1 leaves=12 sample_paths=['bert.encoder.albert_layer_groups.0']_

| field | new model | sibling |
|---|---|---|

### `up_sample1d` — NEW
_module-tree: occ=11 leaves=11 sample_paths=['predictor.F0.0.upsample', 'predictor.F0.1.upsample']_

| field | new model | sibling |
|---|---|---|

### `albert_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=1 leaves=7 sample_paths=['bert.encoder.albert_layer_groups.0.albert_layers.0.attention']_

| field | new model | sibling |
|---|---|---|

### `duration_encoder` — NEW
_module-tree: occ=1 leaves=6 sample_paths=['predictor.text_encoder']_

| field | new model | sibling |
|---|---|---|

### `l_s_t_m` — NEW
_module-tree: occ=6 leaves=6 sample_paths=['predictor.text_encoder.lstms.0', 'predictor.text_encoder.lstms.2']_

| field | new model | sibling |
|---|---|---|

### `albert_embeddings` — NEW
_module-tree: occ=1 leaves=5 sample_paths=['bert.embeddings']_

| field | new model | sibling |
|---|---|---|

### `ada_layer_norm` — NEW
_module-tree: occ=3 leaves=3 sample_paths=['predictor.text_encoder.lstms.1', 'predictor.text_encoder.lstms.3']_

| field | new model | sibling |
|---|---|---|

### `source_module_hn_n_s_f` — NEW
_module-tree: occ=1 leaves=3 sample_paths=['decoder.generator.m_source']_

| field | new model | sibling |
|---|---|---|

### `leaky_re_l_u` — NEW
_module-tree: occ=2 leaves=2 sample_paths=['predictor.F0.0.actv', 'text_encoder.cnn.0.2']_

| field | new model | sibling |
|---|---|---|

### `custom_s_t_f_t` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['decoder.generator.stft']_

| field | new model | sibling |
|---|---|---|

### `linear_norm` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['predictor.duration_proj']_

| field | new model | sibling |
|---|---|---|

### `reflection_pad1d` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['decoder.generator.reflection_pad']_

| field | new model | sibling |
|---|---|---|

### `sine_gen` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['decoder.generator.m_source.l_sin_gen']_

| field | new model | sibling |
|---|---|---|

### `upsample` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['decoder.generator.f0_upsamp']_

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
