# Bring-up plan: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`

Backend template: **NemotronH (nemotron_h hybrid Mamba2/MoE)** at `models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16` (canonical HF id: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).
New `model_type` = `nemotron_h`; sibling `model_type` = `nemotron_h`.

**Summary:** 3 REUSE · 5 NEW component(s).

> **Notes:**
> - Top sibling candidates (per-component reuse targets are pulled from whichever sibling provides them, not only the first): NemotronH (nemotron_h hybrid Mamba2/MoE) (score 100: exact model_type 'nemotron_h'); tt_transformers / simple_text_demo (score 40: category 'LLM' default (generic runner)); falcon7b_common (auto-upstream) (score 30: category 'LLM' default)

## Sibling candidates (ranked)

Top backends by match score — components pull their reuse target from whichever of these provides it, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `NemotronH (nemotron_h hybrid Mamba2/MoE)` (selected) | 100 | exact model_type 'nemotron_h' |
| 2 | `tt_transformers / simple_text_demo` | 40 | category 'LLM' default (generic runner) |
| 3 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **NEW** | `nemotron_h_block` | `—` | `transformers/src/transformers/models/nemotron_h/modeling_nemotron_h.py` |
| **ADAPT** | `nemotron_h_mo_e` | `models/tt_transformers/tt/mixtral_moe.py` | `—` |
| **ADAPT** | `nemotron_h_mamba2_mixer` | `models/demos/wormhole/mamba/tt/mamba_ssm.py` | `—` |
| **REUSE** | `nemotron_h_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `nemotron_h_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **NEW** | `re_l_u_squared_activation` | `—` | `transformers/src/transformers/models/nemotron_h/modeling_nemotron_h.py` |
| **REUSE** | `nemotron_h_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `nemotron_h_experts` | `—` | `transformers/src/transformers/models/nemotron_h/modeling_nemotron_h.py` |
| **NEW** | `nemotron_h_topk_router` | `—` | `transformers/src/transformers/models/nemotron_h/modeling_nemotron_h.py` |
| **NEW** | `zamba2_r_m_s_norm_gated` | `—` | `transformers/src/transformers/models/nemotron_h/modeling_nemotron_h.py` |

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

### `nemotron_h_block` — NEW
_module-tree: occ=52 leaves=352 sample_paths=['layers.0', 'layers.1']_

| field | new model | sibling |
|---|---|---|

### `nemotron_h_mo_e` — ADAPT
_reuse_registry: moe_routing_mixtral_style -> models/tt_transformers/tt/mixtral_moe.py::TtMoeLayer (ADAPT). derived from compatibility.py BUILDING_BLOCKS 'MoE routing (Mixtral-style)'. Generic MoE block currently hard-codes num_devices=8 and top-2. Adapting to other top-k or device counts needs a small refactor. Larger-scale MoE (DeepSeek/GPT-OSS) has standalone demos. | module-tree: occ=23 leaves=161 sample_paths=['layers.1.mixer', 'layers.3.mixer']_

| field | new model | sibling |
|---|---|---|

### `nemotron_h_mamba2_mixer` — ADAPT
_reuse_registry: ssm_mamba_blocks -> models/demos/wormhole/mamba/tt/mamba_ssm.py::TtMambaSSM (ADAPT). derived from compatibility.py BUILDING_BLOCKS 'SSM / Mamba blocks'. Separate stack from tt_transformers; covers state-spaces/mamba-2.8b-slimpj. Other Mamba variants would need adaptation. | module-tree: occ=23 leaves=115 sample_paths=['layers.0.mixer', 'layers.2.mixer']_

| field | new model | sibling |
|---|---|---|

### `nemotron_h_m_l_p` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=23 leaves=69 sample_paths=['layers.1.mixer.shared_experts', 'layers.3.mixer.shared_experts']_

| field | new model | sibling |
|---|---|---|

### `nemotron_h_r_m_s_norm` — REUSE
_reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=53 leaves=53 sample_paths=['layers.0.norm', 'layers.1.norm']_

| field | new model | sibling |
|---|---|---|

### `re_l_u_squared_activation` — NEW
_module-tree: occ=46 leaves=46 sample_paths=['layers.1.mixer.experts.act_fn', 'layers.1.mixer.shared_experts.act_fn']_

| field | new model | sibling |
|---|---|---|

### `nemotron_h_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=6 leaves=24 sample_paths=['layers.5.mixer', 'layers.12.mixer']_

| field | new model | sibling |
|---|---|---|

### `nemotron_h_experts` — NEW
_module-tree: occ=23 leaves=23 sample_paths=['layers.1.mixer.experts', 'layers.3.mixer.experts']_

| field | new model | sibling |
|---|---|---|

### `nemotron_h_topk_router` — NEW
_module-tree: occ=23 leaves=23 sample_paths=['layers.1.mixer.gate', 'layers.3.mixer.gate']_

| field | new model | sibling |
|---|---|---|

### `zamba2_r_m_s_norm_gated` — NEW
_module-tree: occ=23 leaves=23 sample_paths=['layers.0.mixer.norm', 'layers.2.mixer.norm']_

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
