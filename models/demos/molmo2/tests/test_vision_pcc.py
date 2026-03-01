# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for Molmo2 Vision Transformer blocks.

Compares TTNN implementation against PyTorch reference.
"""

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
    # Load attention weights
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

    # Reshape to [batch, num_heads, seq_len, head_dim]
    batch_size = hidden_states.shape[0]
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention (no causal mask for ViT)
    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_probs, v)

    # Reshape and output projection
    attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
    attn_out = F.linear(attn_out, out_weight, out_bias)

    # Residual
    after_attn = hidden_states + attn_out

    # Step 3: ffn_norm (pre-MLP LayerNorm)
    ffn_norm = RefLayerNorm(hidden_dim, layer_norm_eps)
    ffn_norm.weight.data = state_dict[f"{prefix}.ffn_norm.weight"]
    ffn_norm.bias.data = state_dict[f"{prefix}.ffn_norm.bias"]
    mlp_normed = ffn_norm(after_attn)

    # Step 4: MLP (standard 2-layer MLP with GELU)
    # Keys: feed_forward.w1/w2
    w1_weight = state_dict[f"{prefix}.feed_forward.w1.weight"]
    w1_bias = state_dict[f"{prefix}.feed_forward.w1.bias"]
    w2_weight = state_dict[f"{prefix}.feed_forward.w2.weight"]
    w2_bias = state_dict[f"{prefix}.feed_forward.w2.bias"]

    # w1 -> GELU -> w2
    hidden = F.linear(mlp_normed, w1_weight, w1_bias)
    hidden = F.gelu(hidden, approximate="tanh")  # QuickGELU
    mlp_out = F.linear(hidden, w2_weight, w2_bias)

    # Final residual
    block_out = after_attn + mlp_out

    return block_out


def test_vit_blocks_pcc():
    """Test PCC for Vision Transformer blocks."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    # Config
    hidden_dim = 1152
    intermediate_dim = 4304
    num_heads = 16
    head_dim = 72
    num_patches = 729  # 27x27 for 378x378 image with 14x14 patches
    layer_norm_eps = 1e-6
    num_layers = 25

    # Load weights
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")
    logger.info("Loaded weights")

    # Create random input (simulating embedded patches)
    torch.manual_seed(42)
    ref_hidden = torch.randn(1, num_patches, hidden_dim)

    # Add positional embedding
    pos_embed = state_dict["model.vision_backbone.image_vit.positional_embedding"]
    ref_hidden = ref_hidden + pos_embed.unsqueeze(0)

    logger.info(f"Input shape: {ref_hidden.shape}")
    logger.info(f"Input stats: min={ref_hidden.min():.4f}, mean={ref_hidden.mean():.4f}, max={ref_hidden.max():.4f}")

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        from models.demos.molmo2.tt.vision_block import VisionBlock

        logger.info("=" * 80)
        logger.info("ViT Block PCC comparison")
        logger.info("=" * 80)

        ttnn_hidden = ref_hidden.clone()

        for layer_num in range(num_layers):  # Test all 25 layers
            # Reference forward
            ref_out = ref_vit_block_forward(
                ref_hidden,
                state_dict,
                layer_num,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                layer_norm_eps=layer_norm_eps,
            )

            # TTNN forward
            ttnn_block = VisionBlock(
                mesh_device=device,
                state_dict=state_dict,
                layer_num=layer_num,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                layer_norm_eps=layer_norm_eps,
                # Don't override state_dict_prefix - let it use the default
                dtype=ttnn.bfloat16,
            )

            hidden_ttnn = ttnn.from_torch(
                ttnn_hidden.unsqueeze(0),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            ttnn_out = ttnn_block(hidden_ttnn)
            ttnn_out_torch = ttnn.to_torch(ttnn_out).squeeze(0).float()

            # Calculate PCC
            pcc, status = calculate_pcc(ref_out, ttnn_out_torch)
            diff = (ref_out - ttnn_out_torch).abs()

            logger.info(
                f"ViT Layer {layer_num:2d}: PCC={pcc:.6f}, max_diff={diff.max():.4f}, mean_diff={diff.mean():.6f}"
            )

            # Update for next layer
            ref_hidden = ref_out
            ttnn_hidden = ttnn_out_torch

            # Clean up
            ttnn.deallocate(hidden_ttnn)
            ttnn.deallocate(ttnn_out)

            if pcc < 0.9:
                logger.warning(f"PCC dropped below 0.9 at layer {layer_num}")
                break

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_vit_blocks_pcc()
