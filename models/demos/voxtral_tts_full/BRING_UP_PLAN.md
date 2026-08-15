# Bring-up plan: `/localdev/lserbedzija/hf_models/voxtral-tts-full`

Backend template: **XTTS-v2 (multilingual TTS)** at `models/demos/xtts_v2` (canonical HF id: `/local/ttuser/apande/models/XTTS-v2-hf`).
New `model_type` = `voxtral_tts`; sibling `model_type` = `None`.

**Summary:** 3 REUSE · 4 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `/local/ttuser/apande/models/XTTS-v2-hf` and re-run for a sharper diff.
> - Top sibling candidates (per-component reuse targets are pulled from whichever sibling provides them, not only the first): hf_eager universal (TTS) (score 40: category 'TTS' default (generic runner)); XTTS-v2 (multilingual TTS) (score 30: category 'TTS' default)

## Sibling candidates (ranked)

Top backends by match score — components pull their reuse target from whichever of these provides it, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `hf_eager universal (TTS)` | 40 | category 'TTS' default (generic runner) |
| 2 | `XTTS-v2 (multilingual TTS)` (selected) | 30 | category 'TTS' default |

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **NEW** | `tts_backbone` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `decoder_layer` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **REUSE** | `r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **NEW** | `codec_decoder` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `flow_matching` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |

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

### `tts_backbone` — NEW
_module-tree: occ=1 leaves=105 sample_paths=['backbone']_

| field | new model | sibling |
|---|---|---|

### `decoder_layer` — NEW
_module-tree: occ=26 leaves=104 sample_paths=['backbone.layers.0', 'backbone.layers.1']_

| field | new model | sibling |
|---|---|---|

### `r_m_s_norm` — REUSE
_reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=53 leaves=53 sample_paths=['backbone.layers.0.input_layernorm', 'backbone.layers.0.post_attention_layernorm']_

| field | new model | sibling |
|---|---|---|

### `attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=26 leaves=26 sample_paths=['backbone.layers.0.self_attn', 'backbone.layers.1.self_attn']_

| field | new model | sibling |
|---|---|---|

### `m_l_p` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=26 leaves=26 sample_paths=['backbone.layers.0.mlp', 'backbone.layers.1.mlp']_

| field | new model | sibling |
|---|---|---|

### `codec_decoder` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['codec']_

| field | new model | sibling |
|---|---|---|

### `flow_matching` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['flow']_

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
