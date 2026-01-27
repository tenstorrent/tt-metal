# Qwen3-VL Reference Implementation

This directory contains reference implementations for the Qwen3-VL vision-language model, including both the official HuggingFace transformers module-based implementation and functional equivalents for easier porting and testing.

## Contents

### `model.py`

The official HuggingFace transformers implementation of Qwen3-VL, automatically generated from `src/transformers/models/qwen3_vl/modular_qwen3_vl.py`. This file contains the complete PyTorch module hierarchy:

**Vision Components:**
- `Qwen3VLVisionMLP` - MLP block for the vision transformer
- `Qwen3VLVisionPatchEmbed` - 3D convolution-based patch embedding (handles temporal dimension)
- `Qwen3VLVisionRotaryEmbedding` - Rotary position embeddings for vision
- `Qwen3VLVisionPatchMerger` - Merges spatial patches with LayerNorm and MLP
- `Qwen3VLVisionAttention` - Multi-head attention with rotary embeddings
- `Qwen3VLVisionBlock` - Transformer block combining attention and MLP
- `Qwen3VLVisionModel` - Complete vision encoder with DeepStack feature extraction

**Text Components:**
- `Qwen3VLTextRotaryEmbedding` - Multi-dimensional RoPE (M-RoPE) for temporal/spatial awareness
- `Qwen3VLTextRMSNorm` - RMS normalization layer
- `Qwen3VLTextAttention` - Grouped-query attention with Q/K normalization
- `Qwen3VLTextMLP` - Gated MLP with SiLU activation
- `Qwen3VLTextDecoderLayer` - Decoder layer combining attention and MLP
- `Qwen3VLTextModel` - Text decoder with DeepStack visual feature integration

**Combined Model:**
- `Qwen3VLModel` - Unified vision-language model
- `Qwen3VLForConditionalGeneration` - Full model with language modeling head

### `functional.py`

Stateless functional implementations of key vision components. These pure functions take weights as explicit parameters, making them easier to port to other frameworks or accelerators.

**Available Functions:**

```python
# Rotary embedding helpers
rotate_half(x)
apply_rotary_pos_emb_vision(q, k, cos, sin)

# Position embedding computation
qwen3_vision_rotary_embedding(seqlen, dim, theta=10000.0, device=None)
qwen3_vl_rot_pos_emb(grid_thw, spatial_merge_size, head_dim)
qwen3_vl_fast_pos_embed_interpolation(grid_thw, num_grid_per_side, pos_embed, spatial_merge_size)

# Preprocessing
qwen3_vision_transformer_preprocess(seq_len, grid_thw, head_dim, spatial_merge_size)
```

### `model.txt`

A text representation of the model's weight structure and tensor shapes. Useful for understanding the model architecture at a glance.

## Key Architecture Details

### Vision Encoder

- **Patch Embedding**: Uses 3D convolution with kernel `[temporal_patch_size, patch_size, patch_size]` to handle video frames
- **Spatial Merge**: Reduces spatial resolution by merging patches (default 2×2)
- **DeepStack**: Extracts intermediate visual features at specified layer indices for early fusion with text

### Text Decoder

- **M-RoPE**: Multi-dimensional rotary position embeddings that encode temporal (T), height (H), and width (W) positions separately
- **DeepStack Integration**: Visual features from intermediate vision layers are added to early decoder hidden states
- **QK Normalization**: Both query and key projections use RMSNorm

### Position Encoding

The model uses an interleaved M-RoPE scheme:
1. Separate position IDs for temporal, height, and width dimensions
2. Frequencies are interleaved as `[THWTHWTHW...TT]` pattern
3. Vision tokens use 2D spatial coordinates; text tokens use sequential positions

## Usage

### Loading the Reference Model

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")
```

### Using Functional Implementations

```python
from functional import (
    qwen3_vision_transformer_preprocess,
    qwen3_vl_rot_pos_emb,
)

# Compute rotary embeddings for vision
grid_thw = torch.tensor([[1, 28, 28]])  # 1 frame, 28x28 patches
rotary_emb = qwen3_vl_rot_pos_emb(grid_thw, spatial_merge_size=2, head_dim=80)

# Get preprocessing outputs
cu_seqlens, position_embeddings = qwen3_vision_transformer_preprocess(
    seq_len=784,
    grid_thw=grid_thw,
    head_dim=80,
    spatial_merge_size=2,
)
```

## References

- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
