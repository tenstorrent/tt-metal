# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Full PCC test for Molmo2 Vision Backbone.

Tests the complete vision pipeline against PyTorch reference:
1. Patch embedding + positional embedding
2. ViT encoding (all 25 layers)
3. Multi-scale feature extraction (layers 18 and 24)
4. Feature concatenation
5. Feature gathering using pooled_patches_idx
6. Cross-attention pooling
7. SwiGLU projection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import ttnn


def calculate_pcc(ref, out):
    """Calculate Pearson Correlation Coefficient."""
    if ref.shape != out.shape:
        return -1.0, f"Shape mismatch: {ref.shape} vs {out.shape}"
    ref_flat = ref.flatten().float()
    out_flat = out.flatten().float()
    ref_mean = ref_flat.mean()
    out_mean = out_flat.mean()
    numerator = ((ref_flat - ref_mean) * (out_flat - out_mean)).sum()
    denominator = torch.sqrt(((ref_flat - ref_mean) ** 2).sum() * ((out_flat - out_mean) ** 2).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0, "zero std"
    pcc = numerator / denominator
    return pcc.item(), "ok"


class RefLayerNorm(nn.Module):
    """Reference LayerNorm implementation."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.weight + self.bias


def ref_vit_block_forward(
    hidden_states,
    state_dict,
    layer_num,
    hidden_dim=1152,
    intermediate_dim=4304,
    num_heads=16,
    head_dim=72,
    layer_norm_eps=1e-6,
):
    """Reference PyTorch implementation of a single ViT block."""
    seq_len = hidden_states.shape[1]
    prefix = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}"

    # Step 1: attention_norm (pre-attention LayerNorm)
    attn_norm = RefLayerNorm(hidden_dim, layer_norm_eps)
    attn_norm.weight.data = state_dict[f"{prefix}.attention_norm.weight"]
    attn_norm.bias.data = state_dict[f"{prefix}.attention_norm.bias"]
    normed = attn_norm(hidden_states)

    # Step 2: Attention
    q_weight = state_dict[f"{prefix}.attention.wq.weight"]
    k_weight = state_dict[f"{prefix}.attention.wk.weight"]
    v_weight = state_dict[f"{prefix}.attention.wv.weight"]
    out_weight = state_dict[f"{prefix}.attention.wo.weight"]

    q_bias = state_dict[f"{prefix}.attention.wq.bias"]
    k_bias = state_dict[f"{prefix}.attention.wk.bias"]
    v_bias = state_dict[f"{prefix}.attention.wv.bias"]
    out_bias = state_dict[f"{prefix}.attention.wo.bias"]

    # Q, K, V projections
    q = F.linear(normed, q_weight, q_bias)
    k = F.linear(normed, k_weight, k_bias)
    v = F.linear(normed, v_weight, v_bias)

    # Reshape
    batch_size = hidden_states.shape[0]
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Attention
    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_probs, v)

    # Output
    attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
    attn_out = F.linear(attn_out, out_weight, out_bias)

    # Residual
    after_attn = hidden_states + attn_out

    # MLP
    ffn_norm = RefLayerNorm(hidden_dim, layer_norm_eps)
    ffn_norm.weight.data = state_dict[f"{prefix}.ffn_norm.weight"]
    ffn_norm.bias.data = state_dict[f"{prefix}.ffn_norm.bias"]
    mlp_normed = ffn_norm(after_attn)

    w1_weight = state_dict[f"{prefix}.feed_forward.w1.weight"]
    w1_bias = state_dict[f"{prefix}.feed_forward.w1.bias"]
    w2_weight = state_dict[f"{prefix}.feed_forward.w2.weight"]
    w2_bias = state_dict[f"{prefix}.feed_forward.w2.bias"]

    hidden = F.linear(mlp_normed, w1_weight, w1_bias)
    hidden = F.gelu(hidden, approximate="tanh")
    mlp_out = F.linear(hidden, w2_weight, w2_bias)

    block_out = after_attn + mlp_out
    return block_out


def ref_image_pooling_forward(
    query,  # [B*N_out, 1, input_dim]
    key_value,  # [B*N_out, K_pool, input_dim]
    state_dict,
    attn_mask=None,  # [B*N_out, 1, 1, K_pool]
    input_dim=2304,
    hidden_dim=1152,
    num_heads=16,
    head_dim=72,
):
    """Reference PyTorch implementation of image pooling cross-attention."""
    prefix = "model.vision_backbone.image_pooling_2d"

    batch_n_out = query.shape[0]
    pool_size = key_value.shape[1]

    wq = state_dict[f"{prefix}.wq.weight"]
    bq = state_dict[f"{prefix}.wq.bias"]
    wk = state_dict[f"{prefix}.wk.weight"]
    bk = state_dict[f"{prefix}.wk.bias"]
    wv = state_dict[f"{prefix}.wv.weight"]
    bv = state_dict[f"{prefix}.wv.bias"]
    wo = state_dict[f"{prefix}.wo.weight"]
    bo = state_dict[f"{prefix}.wo.bias"]

    q = F.linear(query, wq, bq)
    k = F.linear(key_value, wk, bk)
    v = F.linear(key_value, wv, bv)

    q = q.reshape(batch_n_out, 1, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_n_out, pool_size, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_n_out, pool_size, num_heads, head_dim).transpose(1, 2)

    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    if attn_mask is not None:
        attn_weights = attn_weights.masked_fill(~attn_mask.bool(), float("-inf"))

    attn_probs = F.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_probs, v)

    attn_out = attn_out.transpose(1, 2).reshape(batch_n_out, 1, hidden_dim)
    output = F.linear(attn_out, wo, bo)

    return output


def ref_image_projector_forward(x, state_dict):
    """Reference PyTorch implementation of image projector (SwiGLU)."""
    prefix = "model.vision_backbone.image_projector"

    w1 = state_dict[f"{prefix}.w1.weight"]
    w2 = state_dict[f"{prefix}.w2.weight"]
    w3 = state_dict[f"{prefix}.w3.weight"]

    gate = F.silu(F.linear(x, w1))
    up = F.linear(x, w3)
    hidden = gate * up
    output = F.linear(hidden, w2)

    return output


def ref_full_vision_backbone(
    pixel_values,  # [B, 3, H, W]
    pooled_patches_idx,  # [B, N_out, K_pool]
    state_dict,
    patch_size=14,
    hidden_dim=1152,
    num_layers=25,
    feature_layers=(24, 18),  # HF order: [-3, -9]
):
    """Reference full vision backbone."""
    batch_size = pixel_values.shape[0]
    height = pixel_values.shape[2]
    width = pixel_values.shape[3]
    patches_h = height // patch_size
    patches_w = width // patch_size
    num_patches = patches_h * patches_w

    logger.info(f"Input: batch={batch_size}, patches={num_patches}")

    # 1. Patch embedding
    patch_weight = state_dict["model.vision_backbone.image_vit.patch_embedding.weight"]
    patch_bias = state_dict["model.vision_backbone.image_vit.patch_embedding.bias"]

    # Unfold patches
    x = pixel_values.unfold(2, patch_size, patch_size)
    x = x.unfold(3, patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, num_patches, 3 * patch_size * patch_size)

    # Linear projection
    patch_weight_linear = patch_weight.reshape(hidden_dim, -1).transpose(-2, -1)
    x = torch.matmul(x, patch_weight_linear) + patch_bias
    logger.info(f"After patch embed: {x.shape}")

    # 2. Positional embedding
    pos_embed = state_dict["model.vision_backbone.image_vit.positional_embedding"]
    x = x + pos_embed.unsqueeze(0)
    logger.info(f"After pos embed: {x.shape}")

    # 3. ViT encoding
    all_hidden_states = []
    for layer_num in range(num_layers):
        x = ref_vit_block_forward(
            x,
            state_dict,
            layer_num,
            hidden_dim=hidden_dim,
        )
        all_hidden_states.append(x.clone())

    # 4. Multi-scale feature extraction
    features = []
    for layer_idx in feature_layers:
        features.append(all_hidden_states[layer_idx])
    image_features = torch.cat(features, dim=-1)
    logger.info(f"Concatenated features: {image_features.shape}")

    # 5. Gather features
    pool_dim = image_features.shape[-1]
    n_out = pooled_patches_idx.shape[1]
    k_pool = pooled_patches_idx.shape[2]

    valid = pooled_patches_idx >= 0
    valid_token = torch.any(valid, dim=-1)

    batch_idx = torch.arange(batch_size, dtype=torch.long)
    batch_idx = batch_idx.view(batch_size, 1, 1).expand(-1, n_out, k_pool)

    clipped_idx = torch.clip(pooled_patches_idx, min=0)
    to_pool = image_features.reshape(batch_size, -1, pool_dim)[batch_idx, clipped_idx]
    to_pool = to_pool * valid.unsqueeze(-1).float()
    to_pool = to_pool.reshape(-1, k_pool, pool_dim)

    # 6. Query computation
    valid_flat = valid.reshape(-1, k_pool).float()
    denom = valid_flat.sum(-1, keepdim=True)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    query = to_pool.sum(-2, keepdim=True) / denom.unsqueeze(-1)

    # Attention mask
    attn_mask = valid.reshape(-1, 1, 1, k_pool)

    logger.info(f"Query: {query.shape}, to_pool: {to_pool.shape}")

    # 7. Cross-attention pooling
    pooled = ref_image_pooling_forward(query, to_pool, state_dict, attn_mask=attn_mask)
    pooled = pooled.reshape(batch_size, -1, 1152)
    logger.info(f"Pooled features: {pooled.shape}")

    # 8. SwiGLU projection
    projected = ref_image_projector_forward(pooled, state_dict)
    logger.info(f"Projected features: {projected.shape}")

    # 9. Valid token filtering
    final = projected.view(-1, projected.shape[-1])[valid_token.flatten()]
    logger.info(f"Final features: {final.shape}")

    return final


def test_full_vision_pcc():
    """Test full vision backbone PCC."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    logger.info("Loading weights...")
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")
    logger.info("Loaded weights")

    # Create random image input
    torch.manual_seed(42)
    batch_size = 1
    pixel_values = torch.randn(batch_size, 3, 378, 378)

    # Create pooled_patches_idx (simulating 2x2 pooling)
    patches_per_side = 27
    pool_h, pool_w = 2, 2

    # Create patch indices
    idx_arr = np.arange(patches_per_side * patches_per_side).reshape(patches_per_side, patches_per_side)

    # Pad to make divisible by pool size
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        constant_values=-1,
    )

    # Rearrange for pooling
    h_out = idx_arr.shape[0] // pool_h
    w_out = idx_arr.shape[1] // pool_w
    idx_arr = idx_arr.reshape(h_out, pool_h, w_out, pool_w)
    idx_arr = idx_arr.transpose(0, 2, 1, 3).reshape(-1, pool_h * pool_w)

    pooled_patches_idx = torch.from_numpy(idx_arr).long().unsqueeze(0)
    logger.info(f"pooled_patches_idx shape: {pooled_patches_idx.shape}")
    logger.info(f"pooled_patches_idx range: [{pooled_patches_idx.min()}, {pooled_patches_idx.max()}]")

    # Reference forward
    logger.info("=" * 80)
    logger.info("Reference forward")
    logger.info("=" * 80)

    ref_output = ref_full_vision_backbone(
        pixel_values,
        pooled_patches_idx,
        state_dict,
    )

    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(
        f"Reference output stats: min={ref_output.min():.4f}, mean={ref_output.mean():.4f}, max={ref_output.max():.4f}"
    )

    # TTNN forward
    logger.info("=" * 80)
    logger.info("TTNN forward")
    logger.info("=" * 80)

    device = ttnn.open_device(device_id=0)

    try:
        from models.demos.molmo2.tt.vision_backbone import VisionBackbone

        ttnn_backbone = VisionBackbone(
            mesh_device=device,
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
        )

        ttnn_output = ttnn_backbone(
            images_embedded=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
        )

        logger.info(f"TTNN output shape: {ttnn_output.shape}")
        logger.info(
            f"TTNN output stats: min={ttnn_output.min():.4f}, mean={ttnn_output.mean():.4f}, max={ttnn_output.max():.4f}"
        )

        # Compare
        pcc, _ = calculate_pcc(ref_output, ttnn_output)
        diff = (ref_output - ttnn_output).abs()

        logger.info("=" * 80)
        logger.info(f"Full Vision Backbone PCC: {pcc:.6f}")
        logger.info(f"Max diff: {diff.max():.4f}, Mean diff: {diff.mean():.6f}")
        logger.info("=" * 80)

        return pcc

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_full_vision_pcc()
