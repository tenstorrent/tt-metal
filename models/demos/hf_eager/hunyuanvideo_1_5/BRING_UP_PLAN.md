# Bring-up plan: `tencent/HunyuanVideo-1.5`

Backend template: **hf_eager universal (Video)** at `models/demos/hf_eager/demo.py` (canonical HF id: `None`).

**Summary:** 3 REUSE · 18 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `None` and re-run for a sharper diff.

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **REUSE** | `self_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mlp` | `models/tt_transformers/tt/mlp.py` | `—` |
| **NEW** | `hunyuan_video15_transformer_block` | `—` | `—` |
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `hunyuan_video15_token_refiner` | `—` | `—` |
| **NEW** | `hunyuan_video15_individual_token_refiner` | `—` | `—` |
| **NEW** | `hunyuan_video15_individual_token_refiner_block` | `—` | `—` |
| **NEW** | `feed_forward` | `—` | `—` |
| **NEW** | `ada_layer_norm_zero` | `—` | `—` |
| **NEW** | `combined_timestep_text_proj_embeddings` | `—` | `—` |
| **NEW** | `timestep_embedding` | `—` | `—` |
| **NEW** | `hunyuan_video15_by_t5_text_projection` | `—` | `—` |
| **NEW** | `hunyuan_video15_image_projection` | `—` | `—` |
| **NEW** | `hunyuan_video15_ada_norm` | `—` | `—` |
| **NEW** | `hunyuan_video15_time_embedding` | `—` | `—` |
| **NEW** | `linear_activation` | `—` | `—` |
| **NEW** | `ada_layer_norm_continuous` | `—` | `—` |
| **NEW** | `pix_art_alpha_text_projection` | `—` | `—` |
| **NEW** | `timesteps` | `—` | `—` |
| **NEW** | `hunyuan_video15_patch_embed` | `—` | `—` |
| **NEW** | `hunyuan_video15_rotary_pos_embed` | `—` | `—` |

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

### `hunyuan_video15_transformer_block` — NEW
_[supplemental module-tree pass] module-tree: occ=2 leaves=54 sample_paths=['transformer_blocks.0', 'transformer_blocks.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=4 leaves=36 sample_paths=['context_embedder.token_refiner.refiner_blocks.0.attn', 'context_embedder.token_refiner.refiner_blocks.1.attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `hunyuan_video15_token_refiner` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=34 sample_paths=['context_embedder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `hunyuan_video15_individual_token_refiner` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=26 sample_paths=['context_embedder.token_refiner'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `hunyuan_video15_individual_token_refiner_block` — NEW
_[supplemental module-tree pass] module-tree: occ=2 leaves=26 sample_paths=['context_embedder.token_refiner.refiner_blocks.0', 'context_embedder.token_refiner.refiner_blocks.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `feed_forward` — NEW
_[supplemental module-tree pass] module-tree: occ=6 leaves=20 sample_paths=['context_embedder.token_refiner.refiner_blocks.0.ff', 'context_embedder.token_refiner.refiner_blocks.1.ff'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ada_layer_norm_zero` — NEW
_[supplemental module-tree pass] module-tree: occ=4 leaves=12 sample_paths=['transformer_blocks.0.norm1', 'transformer_blocks.0.norm1_context'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `combined_timestep_text_proj_embeddings` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=7 sample_paths=['context_embedder.time_text_embed'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `timestep_embedding` — NEW
_[supplemental module-tree pass] module-tree: occ=2 leaves=6 sample_paths=['context_embedder.time_text_embed.timestep_embedder', 'time_embed.timestep_embedder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `hunyuan_video15_by_t5_text_projection` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=5 sample_paths=['context_embedder_2'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `hunyuan_video15_image_projection` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=5 sample_paths=['image_embedder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `hunyuan_video15_ada_norm` — NEW
_[supplemental module-tree pass] module-tree: occ=2 leaves=4 sample_paths=['context_embedder.token_refiner.refiner_blocks.0.norm_out', 'context_embedder.token_refiner.refiner_blocks.1.norm_out'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `hunyuan_video15_time_embedding` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=4 sample_paths=['time_embed'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `linear_activation` — NEW
_[supplemental module-tree pass] module-tree: occ=2 leaves=4 sample_paths=['context_embedder.token_refiner.refiner_blocks.0.ff.net.0', 'context_embedder.token_refiner.refiner_blocks.1.ff.net.0'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ada_layer_norm_continuous` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=3 sample_paths=['norm_out'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `pix_art_alpha_text_projection` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=3 sample_paths=['context_embedder.time_text_embed.text_embedder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `timesteps` — NEW
_[supplemental module-tree pass] module-tree: occ=2 leaves=2 sample_paths=['context_embedder.time_text_embed.time_proj', 'time_embed.time_proj'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `hunyuan_video15_patch_embed` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['x_embedder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `hunyuan_video15_rotary_pos_embed` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['rope'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
