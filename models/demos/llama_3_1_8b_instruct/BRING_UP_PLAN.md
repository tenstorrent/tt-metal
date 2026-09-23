# Bring-up plan: `/home/ttuser/benchmark-data/Llama-3.1-8B-Instruct`

Backend template: **NemotronH (nemotron_h hybrid Mamba2/MoE)** at `models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16` (canonical HF id: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).
New `model_type` = `llama`; sibling `model_type` = `None`.

**Summary:** 0 REUSE · 1 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` and re-run for a sharper diff.
> - Top sibling candidates (per-component reuse targets are pulled from whichever sibling provides them, not only the first): NemotronH (nemotron_h hybrid Mamba2/MoE) (score 30: category 'LLM' default); falcon7b_common (auto-upstream) (score 30: category 'LLM' default); gemma4 (auto-upstream) (score 30: category 'LLM' default)

## Sibling candidates (ranked)

Top backends by match score — components pull their reuse target from whichever of these provides it, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `NemotronH (nemotron_h hybrid Mamba2/MoE)` (selected) | 30 | category 'LLM' default |
| 2 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |
| 3 | `gemma4 (auto-upstream)` | 30 | category 'LLM' default |

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **NEW** | `decoder_layer` | `—` | `transformers/src/transformers/models/llama/modeling_llama.py` |
| **ADAPT** | `attention` | `models/common/modules/attention/attention_1d.py` | `—` |
| **ADAPT** | `m_l_p` | `models/common/modules/mlp/mlp_1d.py` | `—` |
| **ADAPT** | `r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **ADAPT** | `rotary_embedding` | `models/common/modules/rope/rope_1d.py` | `—` |

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

### `decoder_layer` — NEW
_module-tree: occ=32 leaves=320 sample_paths=['layers.0', 'layers.1']_

| field | new model | sibling |
|---|---|---|

### `attention` — ADAPT
_reuse_registry: gqa_attention -> models/common/modules/attention/attention_1d.py::Attention (ADAPT). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. A generic TT attention module is available and requires adaptation. Complete Llama attention implementations are excluded. | module-tree: occ=32 leaves=128 sample_paths=['layers.0.self_attn', 'layers.1.self_attn']_

| field | new model | sibling |
|---|---|---|

### `m_l_p` — ADAPT
_reuse_registry: swiglu_mlp -> models/common/modules/mlp/mlp_1d.py::MLP (ADAPT). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. Adapt the generic TT MLP module without consulting a complete Llama implementation. | module-tree: occ=32 leaves=128 sample_paths=['layers.0.mlp', 'layers.1.mlp']_

| field | new model | sibling |
|---|---|---|

### `r_m_s_norm` — ADAPT
_reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (ADAPT). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. Adapt the generic TT RMSNorm module and add multi-device behavior as needed. | module-tree: occ=65 leaves=65 sample_paths=['layers.0.input_layernorm', 'layers.0.post_attention_layernorm']_

| field | new model | sibling |
|---|---|---|

### `rotary_embedding` — ADAPT
_reuse_registry: llama_3_rope_scaling -> models/common/modules/rope/rope_1d.py::RotaryEmbedding (ADAPT). Generic TT RoPE building block for the restricted Llama experiment. | module-tree: occ=1 leaves=1 sample_paths=['rotary_emb']_

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
