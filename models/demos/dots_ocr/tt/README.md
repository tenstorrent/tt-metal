# Dots-OCR TTNN vision (`dots_visionTT`)

This folder hosts a **TTNN** port of the Dots vision tower (`DotsVisionTransformer`), aligned with the PyTorch reference under `../reference/dots_ocr/` and `../reference/dots_ocr/config.json`.

## Contents

| File | Purpose |
|------|---------|
| `dots_visionTT.py` | `DotsVisionTransformerTT` — norms, MLP, merger, and QKV/proj on device; patch + RoPE indices + attention core (RoPE + SDPA) match reference numerics. |
| `__init__.py` | Package export: `DotsVisionTransformerTT`. |

## Architecture (high level)

1. **Patch embedding** — PyTorch `DotsViTPreprocessor` from the reference, with weights loaded from your checkpoint (`vision_tower.patch_embed.*`).
2. **RoPE positions** — `DotsVisionTransformer.rot_pos_emb` from a reference instance (geometry only; no dependence on trained conv weights).
3. **Trunk** — Per layer on **Wormhole**:
   - `models.common.rmsnorm.RMSNorm` for `norm1` / `norm2` (same family as Qwen2.5-VL vision).
   - Fused **QKV** and **proj** as `ttnn.linear`; **RoPE + block-diagonal attention + SDPA** in **PyTorch** (same pattern as reference `VisionSdpaAttention`) for mask correctness and simpler bring-up.
   - **SwiGLU** MLP (`fc1` / `fc3` SiLU gate × `fc3`, then `fc2`) matching `DotsSwiGLUFFN`.
4. **Post trunk** — Optional `post_trunk_norm` (RMSNorm) when `post_norm` is true in config.
5. **Patch merger** — LayerNorm on `ln_q` (default when `pre_norm` is omitted in JSON), then GELU MLP; layout follows Qwen3-VL `PatchMerger` reshape pattern (`spatial_merge_size²` tokens per row).

## Requirements

- **Target**: Wormhole (compute kernels use `WormholeComputeKernelConfig`).
- **Python deps**: Same stack as other `models/` demos (including `ttnn`, `torch`, and transitive imports such as `loguru` via `tt_transformers` / `models.common`).
- **Checkpoint**: Hugging Face–style keys under a prefix such as `vision_tower.` (see below).

## Constraints

- **Sequence length** for the merger: total patch tokens `S` must be divisible by `spatial_merge_size ** 2` (e.g. `4` when `spatial_merge_size` is `2`).
- **TTNN padding**: The trunk pads `S` up to the next multiple of **128** for matmul/layout; only the first `S` tokens are read back on host after the merger.

## Usage

```python
import torch
from models.demos.dots_ocr.reference.dots_ocr.configuration_dots import DotsVisionConfig
from models.demos.dots_ocr.tt import DotsVisionTransformerTT

# vision_config: DotsVisionConfig or dict matching config.json "vision_config"
vision_config = DotsVisionConfig(...)  # or dict from json.load(...)

# state_dict: full model state_dict; prefix matches your keys (often "vision_tower.")
model = DotsVisionTransformerTT(
    vision_config=vision_config,
    mesh_device=mesh_device,
    state_dict=state_dict,
    state_dict_prefix="vision_tower.",
    weight_cache_path=None,  # optional pathlib.Path for ttnn weight cache
)

# pixel_values / grid_thw: same contract as reference Dots vision forward
out_torch = model(pixel_values, grid_thw, return_host_torch=True)
# out_torch: [num_merged_patches, hidden_size] (bf16 on CPU)
```

To keep output on device:

```python
out_tt = model(pixel_values, grid_thw, return_host_torch=False)
```

## State dict prefix

If keys look like:

- `vision_tower.blocks.0.attn.qkv.weight`
- `vision_tower.merger.ln_q.weight`

use `state_dict_prefix="vision_tower."`.

If the vision tower is nested differently, adjust the prefix so that `f"{prefix}blocks.0.norm1.weight"` resolves correctly.

## Relation to Qwen VL demos

This module intentionally **reuses** building blocks and conventions from:

- **Qwen2.5-VL** — RMSNorm + SwiGLU-style MLP on device.
- **Qwen3-VL** — LayerNorm + two-line GELU merger and reshape/workaround patterns for tilized tensors.

It does **not** duplicate the full Qwen `VisionModelArgs` / demo harness; it is scoped to Dots-OCR vision only.

## PCC test (reference vs TT)

With local HF weights under `reference/dots_ocr/` (or `DOTS_OCR_MODEL_PATH`), run from repo root:

```bash
pytest models/demos/dots_ocr/tests/test_dots_vision_tt_pcc.py -v
```

- Uses the same `mesh_device` / `device_params` pattern as other VL demos (`MESH_DEVICE`, fabric on multi-chip).
- Skips if `config.json` + `model-*-of-*.safetensors` are missing.
- Optional: `reference/test12.png` + processor (same flow as `reference/demo.py`); otherwise synthetic `pixel_values` / `grid_thw`.
- Override minimum PCC with `DOTS_VISION_PCC_REQUIRED` (default `0.97`).

## Limitations / follow-ups

- **Attention**: RoPE + SDPA run in PyTorch between ttnn QKV and ttnn output projection (correctness and `cu_seqlens` block masks first). A full ttnn RoPE + `ttnn.transformer.scaled_dot_product_attention` path can be added later for performance.
- **Multi-image batches**: Supported via `cu_seqlens` in the torch SDPA path (same idea as reference `VisionSdpaAttention`).

## See also

- Reference model: `../reference/dots_ocr/modeling_dots_vision.py`
- Vision config: `../reference/dots_ocr/configuration_dots.py` (`DotsVisionConfig`)
