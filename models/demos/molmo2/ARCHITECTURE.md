# Molmo2-8B Architecture Analysis

## Model Family
Vision-Language Model (VLM): Visual Transformer encoder + cross-attention adapter + GQA decoder-only LLM.

## Component Inventory

| Component | Weight Prefix | Tensor Count (approx) | Required For | Implementation Status |
|-----------|--------------|----------------------|--------------|----------------------|
| ViT Encoder (25 layers) | `model.vision_backbone.image_vit.*` | ~300 | Image feature extraction | Done |
| Image Pooling (cross-attn) | `model.vision_backbone.image_pooling_2d.*` | 8 | Multi-scale feature pooling | Done |
| Image Projector (SwiGLU) | `model.vision_backbone.image_projector.*` | 3 | ViT→LM dimension projection | Done |
| Text Decoder (36 layers) | `model.transformer.blocks.*` | ~432 | Token generation | Done |
| Token Embedding | `model.transformer.wte.*` | 2 | Text/image token embeddings | Done |
| LM Head | `lm_head.weight` | 1 | Logit projection | Done |

Total: ~8.66B parameters. All components implemented and tested.

## Architecture Data Flow

```
Image (378×378)
    ↓ patch_embedding (14×14 patches → 729 tokens, linear+bias)
    ↓ + learned positional embedding [729, 1152]
ViT layers 1–25  [383M, LayerNorm, GELU, separate wq/wk/wv/wo+bias]
    ↓ collect hidden states at layers 24 and 18 (HF: vit_layers=[-3, -9])
    ↓ concat on hidden dim: [B*T, 729, 1152] × 2 → [B*T, 729, 2304]
image_pooling_2d [9.3M]
    cross-attention: Q=masked mean over pooled neighborhoods
    KV=gathered patch features via pooled_patches_idx
    wq/wk/wv: 2304→1152, wo: 1152→1152 (all with bias)
    ↓ [B, N_out, 1152]
image_projector [78.6M]
    SwiGLU: w1/w3: 1152→12288, w2: 12288→4096 (no bias)
    ↓ [valid_tokens, 4096]
splice into token sequence at image_patch_id (151938) positions
Language Model [8,192M, 36 layers, GQA 32/8, QK-norm (qwen3), RoPE θ=1M]
    ↓
Logits [vocab=152064]
```

## Similar Implementations

| Component | Reference Implementation | Similarity |
|-----------|-------------------------|------------|
| ViT Attention | `models/demos/vision/classification/vit/common/tt/ttnn_functional_vit.py` | MHA with separate QKV+bias, no causal mask |
| Image Pooling | `models/tt_transformers/tt/multimodal/llama_cross_attention.py` | Cross-attention with gather via pooled_patches_idx |
| SwiGLU Projector | `models/tt_transformers/tt/mlp.py` | SwiGLU (w1/w3 gate+up, w2 down) |
| Text Attention | `models/demos/qwen3_vl/tt/` | GQA with QK-norm (qwen3-style) |
| RMSNorm | `models/common/rmsnorm.py` | Standard RMSNorm |
| RoPE | `models/demos/llama3_70b_galaxy/tt/llama_rope.py` | RoPE with θ=1M |
| Generator | `models/demos/qwen3_vl/tt/generator.py` | Vision token splicing pattern |

## Key Differences from Reference Implementations

1. **ViT uses head_dim=72** (not 64 or 128): requires padding to 96 for TILE_LAYOUT
2. **Multi-scale feature concat**: ViT returns hidden states from layers 18 AND 24, concatenated on the last dim to form 2304-dim features (not just the final layer)
3. **pooled_patches_idx gather**: Cross-attention pooling uses a precomputed index tensor from the image preprocessor — not standard self-attention pooling
4. **QK-norm type "qwen3"**: Per-head RMSNorm on Q and K vectors (qwen3 style), unlike standard per-projection QK-norm
5. **Feature layer order (24, 18)**: Matches HF `vit_layers=[-3, -9]` ordering; concat is `cat([layer_24_out, layer_18_out], dim=-1)`. Deviating from this silently produces wrong results.
6. **Extended vocabulary**: vocab_size=152064 = 151936 base + 128 special (including image_patch_id=151938)
7. **Fused QKV for text decode, separate for prefill**: Decode path uses single fused `wqkv` matmul; prefill uses 3 separate projections

## Weight Mapping

### ViT Block (layer N)

| HuggingFace Key | TTNN/Code Key |
|----------------|---------------|
| `model.vision_backbone.image_vit.transformer.resblocks.{N}.attention.wq.{weight,bias}` | `{prefix}.wq.{weight,bias}` |
| `model.vision_backbone.image_vit.transformer.resblocks.{N}.attention.wk.{weight,bias}` | `{prefix}.wk.{weight,bias}` |
| `model.vision_backbone.image_vit.transformer.resblocks.{N}.attention.wv.{weight,bias}` | `{prefix}.wv.{weight,bias}` |
| `model.vision_backbone.image_vit.transformer.resblocks.{N}.attention.wo.{weight,bias}` | `{prefix}.wo.{weight,bias}` |
| `model.vision_backbone.image_vit.transformer.resblocks.{N}.attention_norm.{weight,bias}` | `{prefix}.attention_norm.{weight,bias}` |
| `model.vision_backbone.image_vit.transformer.resblocks.{N}.ffn_norm.{weight,bias}` | `{prefix}.ffn_norm.{weight,bias}` |
| `model.vision_backbone.image_vit.transformer.resblocks.{N}.feed_forward.w1.{weight,bias}` | `{prefix}.feed_forward.w1.{weight,bias}` |
| `model.vision_backbone.image_vit.transformer.resblocks.{N}.feed_forward.w2.{weight,bias}` | `{prefix}.feed_forward.w2.{weight,bias}` |

### Vision Adapter

| HuggingFace Key | TTNN/Code Key |
|----------------|---------------|
| `model.vision_backbone.image_pooling_2d.{wq,wk,wv,wo}.{weight,bias}` | `image_pooling_2d.{wq,wk,wv,wo}.{weight,bias}` |
| `model.vision_backbone.image_projector.{w1,w2,w3}.weight` | `image_projector.{w1,w2,w3}.weight` |

### Text Decoder (layer N)

| HuggingFace Key | TTNN/Code Key |
|----------------|---------------|
| `model.transformer.blocks.{N}.self_attn.att_proj.weight` | fused QKV `wqkv.weight` (prefill: split to `wq`, `wk`, `wv`) |
| `model.transformer.blocks.{N}.self_attn.attn_out.weight` | `wo.weight` |
| `model.transformer.blocks.{N}.self_attn.q_norm.weight` | `q_norm_weight` |
| `model.transformer.blocks.{N}.self_attn.k_norm.weight` | `k_norm_weight` |
| `model.transformer.blocks.{N}.mlp.ff_proj.weight` | fused gate+up `w13.weight` |
| `model.transformer.blocks.{N}.mlp.ff_out.weight` | `w2.weight` |
| `model.transformer.blocks.{N}.attn_norm.weight` | `attention_norm.weight` |
| `model.transformer.blocks.{N}.ff_norm.weight` | `ffn_norm.weight` |
| `model.transformer.wte.embedding` | `wte.weight` (base vocab) |
| `model.transformer.wte.new_embedding` | `wte.new_embedding` (extended vocab) |
| `model.transformer.ln_f.weight` | `norm.weight` |
| `lm_head.weight` | `lm_head.weight` |

## Normalization and Positional Encoding

- **Normalization**: Pre-norm (RMSNorm before attention and MLP for text; LayerNorm with bias for ViT)
- **Positional Encoding**: Learned absolute pos embeddings for ViT (bicubic interpolated if image size differs from training); RoPE (θ=1M) for text
- **QK-Norm**: Per-head qwen3-style RMSNorm on Q and K in text attention

## Implementation Order (Completed)

1. Reference model wrapper (`reference/model.py`)
2. ViT block: LayerNorm + VisionAttention + VisionMLP
3. ViT transformer: patch embedding + pos embedding + 25 blocks
4. Image pooling: cross-attention with pooled_patches_idx gather
5. Image projector: SwiGLU 1152→12288→4096
6. Vision backbone: ViT + pooling + projector + feature concat
7. Text attention: GQA + QK-norm + RoPE + KV cache
8. Text MLP: SwiGLU 4096→12288→4096
9. Text decoder: 36 layers + embedding + RMSNorm + LM head
10. Full VLM: vision backbone + token splicing + text decoder
11. Demo + tracing: vision trace + prefill trace + decode trace
