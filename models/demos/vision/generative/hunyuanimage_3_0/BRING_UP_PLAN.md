# Bring-up plan: `tencent/HunyuanImage-3.0`

Backend template: **Stable Diffusion 1.4** at `models/demos/vision/generative/stable_diffusion` (canonical HF id: `CompVis/stable-diffusion-v1-4`).
New `model_type` = `Hunyuan`; sibling `model_type` = `None`.

**Summary:** 5 REUSE · 2 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `CompVis/stable-diffusion-v1-4` and re-run for a sharper diff.

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **REUSE** | `self_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mlp` | `models/tt_transformers/tt/mlp.py` | `—` |
| **NEW** | `image3_decoder_layer` | `—` | `transformers/src/transformers/models/Hunyuan/modeling_Hunyuan.py` |
| **ADAPT** | `mo_e` | `models/tt_transformers/tt/mixtral_moe.py` | `—` |
| **REUSE** | `m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `image3_s_d_p_a_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `top_k_gate` | `—` | `transformers/src/transformers/models/Hunyuan/modeling_Hunyuan.py` |

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
| hidden_act | silu | — |
| hidden_size | 4096 | — |
| intermediate_size | 3072 | — |
| max_position_embeddings | 22800 | — |
| num_attention_heads | 32 | — |
| num_hidden_layers | 32 | — |
| num_key_value_heads | 8 | — |
| patch_size | 1 | — |
| vocab_size | 133120 | — |

### `mlp` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 4096 | — |
| intermediate_size | 3072 | — |
| max_position_embeddings | 22800 | — |
| num_attention_heads | 32 | — |
| num_hidden_layers | 32 | — |
| num_key_value_heads | 8 | — |
| patch_size | 1 | — |
| vocab_size | 133120 | — |

### `image3_decoder_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=32 leaves=6464 sample_paths=['layers.0', 'layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `mo_e` — ADAPT
_[supplemental module-tree pass] reuse_registry: moe_routing_mixtral_style -> models/tt_transformers/tt/mixtral_moe.py::TtMoeLayer (ADAPT). derived from compatibility.py BUILDING_BLOCKS 'MoE routing (Mixtral-style)'. Generic MoE block currently hard-codes num_devices=8 and top-2. Adapting to other top-k or device counts needs a small refactor. Larger-scale MoE (DeepSeek/GPT-OSS) has standalone demos. | module-tree: occ=32 leaves=6272 sample_paths=['layers.0.mlp', 'layers.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=2080 leaves=6240 sample_paths=['layers.0.mlp.shared_mlp', 'layers.0.mlp.experts.0'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=129 leaves=129 sample_paths=['layers.0.self_attn.query_layernorm', 'layers.0.self_attn.key_layernorm'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `image3_s_d_p_a_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=32 leaves=128 sample_paths=['layers.0.self_attn', 'layers.1.self_attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `top_k_gate` — NEW
_[supplemental module-tree pass] module-tree: occ=32 leaves=32 sample_paths=['layers.0.mlp.gate', 'layers.1.mlp.gate'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
