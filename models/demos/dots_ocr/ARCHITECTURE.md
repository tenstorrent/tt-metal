# Dots OCR TTNN Architecture (Hybrid Vision)

## Overview

**Dots OCR** (`rednote-hilab/dots.mocr`) is built on:
- **Vision**: `DotsVisionTransformer` (patch embed + 42 ViT layers + PatchMerger)
- **Text**: Qwen2-style decoder (`Qwen2ForCausalLM` base)
- **Fusion**: `prepare_inputs_embeds` that scatters vision tokens into text embeddings at `image_token_id` positions

## Vision Stack Strategy (Step 4)

**Constraint**: *No full TTNN implementation of `DotsVisionTransformer`* (patch embed + 42 ViT layers on device).

**Hybrid approach** (chosen for speed/stability on WHLB):

### 1. Host/PyTorch Vision Encoder
- Use HF `model.vision_tower(pixel_values, grid_thw)` directly
- Handles:
  - Patch embedding
  - 42 ViT layers (heavy compute)
  - Initial feature extraction
- Returns: `[N_img_tokens, hidden_size]` vision features

### 2. TTNN Patch Merger (Already Implemented)
- `tt/patch_merger.py` (ported from `qwen25_vl`)
- Input: `[B, 1, S_patch, H]` (reshaped vision features)
- Output: `[B, 1, S_img, H_out]` merged vision tokens
- Uses RMSNorm + 2×Linear(GELU) with spatial merging

### 3. Fusion Layer (Host)
- `tt/fusion.py` + `reference/fusion.py`
- `merge_vision_tokens()`: scatter merged vision tokens into text embeddings at image token positions

## Module Interfaces

### `tt/vision.py`
```python
class VisionEncoder:
    def __init__(self, mesh_device, hf_model=None, model_args=None):
        self.mesh_device = mesh_device
        self.hf_model = hf_model  # for host vision_tower
        self.patch_merger = PatchMerger(...)  # TTNN

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Return merged vision tokens [N_tokens, hidden] for fusion."""
```

### Data Flow
```
pixel_values, grid_thw
        ↓ (HF reference)
vision_tower() → [N_img_tokens, hidden]
        ↓ (reshape)
[B, 1, S_patch, H] → PatchMergerTT (device)
        ↓
[B, 1, S_img, H_out] → merge_vision_tokens() (host)
        ↓
inputs_embeds [B, S, D] with vision tokens inserted
```

## WHLB Optimizations
- Single chip: `MeshShape(1, 1)` (N150/N300)
- DRAM heavy: `DRAM_MEMORY_CONFIG` for weights/activations
- Chunked prefill for long sequences (already in `generator.py`)
- Tile layout compatibility via existing transformer stack

## Next Steps (after approval)
1. Integrate `VisionEncoder` into `tt/model.py`
2. End-to-end test with PCC validation
3. Update demo to support `--backend ttnn`
4. Performance benchmarks

This hybrid design gives us **correctness** (HF vision encoder) + **TTNN acceleration** (patch merger) while staying within the "no full TTNN vision transformer" constraint.
