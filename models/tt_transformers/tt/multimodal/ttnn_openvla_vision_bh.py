# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Blackhole-optimized Vision Encoder for OpenVLA.

Fused DINOv2 + SigLIP architecture:
- DINOv2: 24 layers, 1024 dim, 16 heads, 64 head_dim
- SigLIP: 27 layers, 1152 dim, 16 heads, 72 head_dim (padded to 96)

Key optimizations:
- Manual matmul for attention (optimized approach)
- head_dim padding for SigLIP (72 -> 96 to be multiple of 32)
- L1 memory for intermediate tensors
"""

import numpy as np
import torch

import ttnn

# =============================================================================
# Weight Preprocessing Utils
# =============================================================================


def upchannel_attn_weight_bias(qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads):
    """
    Pad attention weights so head_dim is multiple of 32.

    For SigLIP: head_dim=72 -> pad to 96 (3*32)
    """
    qkv = 3
    head_dim = qkv_weight.shape[0] // (num_heads * qkv)
    is_padding_required = head_dim % 32 != 0

    if is_padding_required:
        # Pad to next multiple of 32
        padded_head_dim = int(np.ceil(head_dim / 32) * 32)
        padded_val = padded_head_dim * num_heads * qkv

        # Reshape to [qkv, heads, head_dim, hidden]
        new_qkv_weight = torch.zeros((padded_val, qkv_weight.shape[1]), dtype=qkv_weight.dtype)
        new_qkv_weight = new_qkv_weight.reshape(qkv, num_heads, padded_head_dim, qkv_weight.shape[1])
        reshaped_qkv_weight = qkv_weight.reshape(qkv, num_heads, head_dim, qkv_weight.shape[1])
        new_qkv_weight[:, :, :head_dim, :] = reshaped_qkv_weight
        new_qkv_weight = new_qkv_weight.reshape(padded_val, qkv_weight.shape[1])

        # Pad bias
        new_qkv_bias = torch.zeros((padded_val,), dtype=qkv_bias.dtype)
        new_qkv_bias = new_qkv_bias.reshape(qkv, num_heads, padded_head_dim)
        reshaped_qkv_bias = qkv_bias.reshape(qkv, num_heads, head_dim)
        new_qkv_bias[:, :, :head_dim] = reshaped_qkv_bias
        new_qkv_bias = new_qkv_bias.reshape(-1)

        # Pad proj weight (input dim changes)
        new_proj_weight = torch.zeros((proj_weight.shape[0], padded_head_dim * num_heads), dtype=proj_weight.dtype)
        new_proj_weight = new_proj_weight.reshape(proj_weight.shape[0], num_heads, padded_head_dim)
        reshaped_proj = proj_weight.reshape(proj_weight.shape[0], num_heads, head_dim)
        new_proj_weight[:, :, :head_dim] = reshaped_proj
        new_proj_weight = new_proj_weight.reshape(proj_weight.shape[0], padded_head_dim * num_heads)

        return new_qkv_weight, new_qkv_bias, new_proj_weight, proj_bias

    return qkv_weight, qkv_bias, proj_weight, proj_bias


def preprocess_patch_embed(weight, bias, device):
    """Preprocess patch embedding conv weights for TTNN."""
    # weight: [out_channels, in_channels, H, W] = [hidden, 3, 14, 14]
    out_channels, in_channels, _, _ = weight.shape
    # Pad to 4 input channels
    pad_value = 4 - in_channels
    preprocessed = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
    # Reshape for linear: [patch_pixels * 4, hidden]
    preprocessed = preprocessed.permute(2, 3, 1, 0)  # [H, W, C, hidden]
    preprocessed = preprocessed.reshape(-1, out_channels)  # [H*W*C, hidden]

    return (
        ttnn.from_torch(preprocessed.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(
            bias.unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
    )


# =============================================================================
# DINOv2 Components (16 heads, 64 head_dim)
# =============================================================================


def dinov2_patch_embeddings(pixel_values, weight, bias, pos_embed, cls_token, reg_token, patch_size=14):
    """DINOv2 patch embedding with CLS + 4 register tokens.

    OpenVLA's pos_embed is [1, 256, 1024] - only for patches, NOT including CLS/REG.
    """
    batch_size = pixel_values.shape[0]

    # Fold image into patches: [B, H, W, 4] -> [B, num_patches, patch_size*patch_size*4]
    pixel_values = ttnn.reshape(pixel_values, (batch_size, 16, patch_size, 16, patch_size, 4))
    pixel_values = ttnn.permute(pixel_values, (0, 1, 3, 2, 4, 5))
    pixel_values = ttnn.reshape(pixel_values, (batch_size, 256, patch_size * patch_size * 4))

    # Convert to TILE_LAYOUT with bfloat8_b for perf
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    # Linear projection - bfloat8_b
    patch_embeds = ttnn.linear(
        pixel_values, weight, bias=bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    # Add position embeddings (pos_embed is [1, 256, 1024], same shape as patch_embeds)
    patch_embeds = ttnn.add(patch_embeds, pos_embed, dtype=ttnn.bfloat8_b)

    # Concat CLS (1 token), 4 REG tokens, patches (256 tokens) -> [1, 261, 1024]
    # For batch_size=1, no need to repeat
    # Convert CLS/REG to bfloat8_b to match patches dtype
    cls_reg = ttnn.concat([cls_token, reg_token], dim=1)  # [1, 5, 1024]
    cls_reg = ttnn.typecast(cls_reg, dtype=ttnn.bfloat8_b)

    embeddings = ttnn.concat([cls_reg, patch_embeds], dim=1)  # [1, 261, 1024]
    return embeddings


def dinov2_attention(hidden_states, qkv_weight, qkv_bias, proj_weight, proj_bias, ls_scale, num_heads=16):
    """DINOv2 attention with LayerScale. Uses bfloat8_b for compute performance."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    head_dim = hidden_dim // num_heads
    scale = 1.0 / (head_dim**0.5)

    # QKV projection - bfloat8_b for perf
    qkv = ttnn.linear(
        hidden_states, qkv_weight, bias=qkv_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    # Split Q, K, V and split heads
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(qkv)

    # Attention scores: Q @ K^T - bfloat8_b
    attn_scores = ttnn.matmul(query, key, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    # Scale
    attn_scores = ttnn.mul(attn_scores, scale)

    # Softmax
    attn_probs = ttnn.softmax_in_place(attn_scores)

    # Context: attn @ V - bfloat8_b
    context = ttnn.matmul(attn_probs, value, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(attn_probs)
    ttnn.deallocate(value)

    # Concatenate heads
    context = ttnn.transformer.concatenate_heads(context, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Output projection - bfloat8_b
    output = ttnn.linear(
        context, proj_weight, bias=proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    ttnn.deallocate(context)

    # LayerScale
    output = ttnn.mul(output, ls_scale)

    return output


def dinov2_mlp(hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias, ls_scale):
    """DINOv2 MLP with LayerScale. Uses bfloat8_b for compute performance."""
    output = ttnn.linear(
        hidden_states, fc1_weight, bias=fc1_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    output = ttnn.gelu(output, fast_and_approximate_mode=True)
    output = ttnn.linear(output, fc2_weight, bias=fc2_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    output = ttnn.mul(output, ls_scale)
    return output


def dinov2_block(hidden_states, params):
    """Single DINOv2 transformer block."""
    # Pre-norm attention
    normed = ttnn.layer_norm(
        hidden_states,
        weight=params["norm1_weight"],
        bias=params["norm1_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    attn_out = dinov2_attention(
        normed,
        params["qkv_weight"],
        params["qkv_bias"],
        params["proj_weight"],
        params["proj_bias"],
        params["ls1_scale"],
    )

    hidden_states = ttnn.add(hidden_states, attn_out)

    # Pre-norm MLP
    normed = ttnn.layer_norm(
        hidden_states,
        weight=params["norm2_weight"],
        bias=params["norm2_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    mlp_out = dinov2_mlp(
        normed,
        params["fc1_weight"],
        params["fc1_bias"],
        params["fc2_weight"],
        params["fc2_bias"],
        params["ls2_scale"],
    )

    hidden_states = ttnn.add(hidden_states, mlp_out)

    return hidden_states


# =============================================================================
# SigLIP Components (16 heads, 72 head_dim -> padded to 96)
# =============================================================================


def siglip_patch_embeddings(pixel_values, weight, bias, pos_embed, patch_size=14):
    """SigLIP patch embedding (no CLS token) with position embeddings. Uses bfloat8_b for perf."""
    batch_size = pixel_values.shape[0]

    # Fold image into patches
    pixel_values = ttnn.reshape(pixel_values, (batch_size, 16, patch_size, 16, patch_size, 4))
    pixel_values = ttnn.permute(pixel_values, (0, 1, 3, 2, 4, 5))
    pixel_values = ttnn.reshape(pixel_values, (batch_size, 256, patch_size * patch_size * 4))

    # Convert to TILE_LAYOUT with bfloat8_b for perf
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    # Linear projection - bfloat8_b
    patch_embeds = ttnn.linear(
        pixel_values, weight, bias=bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    # Add position embeddings [1, 256, 1152]
    patch_embeds = ttnn.add(patch_embeds, pos_embed, dtype=ttnn.bfloat8_b)

    return patch_embeds


def siglip_attention(hidden_states, qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads=16, padded_head_dim=96):
    """SigLIP attention with padded head_dim. Uses bfloat8_b for compute performance."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    scale = 1.0 / (padded_head_dim**0.5)

    # QKV projection (weights already padded) - bfloat8_b
    qkv = ttnn.linear(
        hidden_states, qkv_weight, bias=qkv_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    # Split Q, K, V and split heads (now works because padded_head_dim=96 is multiple of 32)
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(qkv)

    # Attention scores - bfloat8_b
    attn_scores = ttnn.matmul(query, key, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attn_scores = ttnn.mul(attn_scores, scale)
    attn_probs = ttnn.softmax_in_place(attn_scores)

    # Context - bfloat8_b
    context = ttnn.matmul(attn_probs, value, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(attn_probs)
    ttnn.deallocate(value)

    # Concatenate heads
    context = ttnn.transformer.concatenate_heads(context, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Output projection - bfloat8_b
    output = ttnn.linear(
        context, proj_weight, bias=proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    ttnn.deallocate(context)

    return output


def siglip_mlp(hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    """SigLIP MLP (no LayerScale). Uses bfloat8_b for compute performance."""
    output = ttnn.linear(
        hidden_states, fc1_weight, bias=fc1_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    output = ttnn.gelu(output, fast_and_approximate_mode=True)
    output = ttnn.linear(output, fc2_weight, bias=fc2_bias, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    return output


def siglip_block(hidden_states, params):
    """Single SigLIP transformer block."""
    # Pre-norm attention
    normed = ttnn.layer_norm(
        hidden_states,
        weight=params["norm1_weight"],
        bias=params["norm1_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    attn_out = siglip_attention(
        normed,
        params["qkv_weight"],
        params["qkv_bias"],
        params["proj_weight"],
        params["proj_bias"],
    )

    hidden_states = ttnn.add(hidden_states, attn_out)

    # Pre-norm MLP
    normed = ttnn.layer_norm(
        hidden_states,
        weight=params["norm2_weight"],
        bias=params["norm2_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    mlp_out = siglip_mlp(
        normed,
        params["fc1_weight"],
        params["fc1_bias"],
        params["fc2_weight"],
        params["fc2_bias"],
    )

    hidden_states = ttnn.add(hidden_states, mlp_out)

    return hidden_states


# =============================================================================
# Full Vision Encoder
# =============================================================================


class OpenVLAVisionEncoderBH:
    """Blackhole-optimized OpenVLA Vision Encoder."""

    def __init__(self, device, state_dict):
        """
        Initialize with OpenVLA state dict.

        Args:
            device: TTNN device
            state_dict: Full openvla-7b state dict
        """
        self.device = device
        self._preprocess_weights(state_dict)

    def _preprocess_weights(self, state_dict):
        """Preprocess all vision encoder weights for TTNN."""

        # ========================
        # DINOv2 weights
        # ========================
        print("   Preprocessing DINOv2 weights...")

        # Patch embedding
        dinov2_patch_w = state_dict["vision_backbone.featurizer.patch_embed.proj.weight"]
        dinov2_patch_b = state_dict["vision_backbone.featurizer.patch_embed.proj.bias"]
        self.dinov2_patch_weight, self.dinov2_patch_bias = preprocess_patch_embed(
            dinov2_patch_w, dinov2_patch_b, self.device
        )

        # Position embedding - OpenVLA pos_embed is ONLY for patches [1, 256, 1024]
        # (unlike standard DINOv2 which has [1, 261, 1024] including CLS/REG)
        pos_embed = state_dict["vision_backbone.featurizer.pos_embed"]  # [1, 256, 1024]
        self.dinov2_pos_embed = ttnn.from_torch(
            pos_embed.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.dinov2_cls_token = ttnn.from_torch(
            state_dict["vision_backbone.featurizer.cls_token"].to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.dinov2_reg_token = ttnn.from_torch(
            state_dict["vision_backbone.featurizer.reg_token"].to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Final layer norm
        self.dinov2_final_norm_w = ttnn.from_torch(
            state_dict["vision_backbone.featurizer.norm.weight"].unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.dinov2_final_norm_b = ttnn.from_torch(
            state_dict["vision_backbone.featurizer.norm.bias"].unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # DINOv2 blocks
        self.dinov2_blocks = []
        for i in range(24):
            prefix = f"vision_backbone.featurizer.blocks.{i}"

            # QKV weights (concatenate Q, K, V)
            q_w = state_dict[f"{prefix}.attn.qkv.weight"][:1024]
            k_w = state_dict[f"{prefix}.attn.qkv.weight"][1024:2048]
            v_w = state_dict[f"{prefix}.attn.qkv.weight"][2048:]
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)

            q_b = state_dict[f"{prefix}.attn.qkv.bias"][:1024]
            k_b = state_dict[f"{prefix}.attn.qkv.bias"][1024:2048]
            v_b = state_dict[f"{prefix}.attn.qkv.bias"][2048:]
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0)

            block_params = {
                "norm1_weight": ttnn.from_torch(
                    state_dict[f"{prefix}.norm1.weight"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "norm1_bias": ttnn.from_torch(
                    state_dict[f"{prefix}.norm1.bias"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "qkv_weight": ttnn.from_torch(
                    qkv_w.t().contiguous().to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "qkv_bias": ttnn.from_torch(
                    qkv_b.unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "proj_weight": ttnn.from_torch(
                    state_dict[f"{prefix}.attn.proj.weight"].t().contiguous().to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "proj_bias": ttnn.from_torch(
                    state_dict[f"{prefix}.attn.proj.bias"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "ls1_scale": ttnn.from_torch(
                    state_dict[f"{prefix}.ls1.scale_factor"].reshape(1, 1, -1).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "norm2_weight": ttnn.from_torch(
                    state_dict[f"{prefix}.norm2.weight"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "norm2_bias": ttnn.from_torch(
                    state_dict[f"{prefix}.norm2.bias"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "fc1_weight": ttnn.from_torch(
                    state_dict[f"{prefix}.mlp.fc1.weight"].t().contiguous().to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "fc1_bias": ttnn.from_torch(
                    state_dict[f"{prefix}.mlp.fc1.bias"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "fc2_weight": ttnn.from_torch(
                    state_dict[f"{prefix}.mlp.fc2.weight"].t().contiguous().to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "fc2_bias": ttnn.from_torch(
                    state_dict[f"{prefix}.mlp.fc2.bias"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "ls2_scale": ttnn.from_torch(
                    state_dict[f"{prefix}.ls2.scale_factor"].reshape(1, 1, -1).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
            }
            self.dinov2_blocks.append(block_params)

        # ========================
        # SigLIP weights
        # ========================
        print("   Preprocessing SigLIP weights...")

        # Patch embedding
        siglip_patch_w = state_dict["vision_backbone.fused_featurizer.patch_embed.proj.weight"]
        siglip_patch_b = state_dict["vision_backbone.fused_featurizer.patch_embed.proj.bias"]
        self.siglip_patch_weight, self.siglip_patch_bias = preprocess_patch_embed(
            siglip_patch_w, siglip_patch_b, self.device
        )

        # Position embedding [1, 256, 1152]
        siglip_pos_embed = state_dict["vision_backbone.fused_featurizer.pos_embed"]
        self.siglip_pos_embed = ttnn.from_torch(
            siglip_pos_embed.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Final layer norm
        self.siglip_final_norm_w = ttnn.from_torch(
            state_dict["vision_backbone.fused_featurizer.norm.weight"].unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.siglip_final_norm_b = ttnn.from_torch(
            state_dict["vision_backbone.fused_featurizer.norm.bias"].unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # SigLIP blocks (with head_dim padding)
        self.siglip_blocks = []
        for i in range(27):
            prefix = f"vision_backbone.fused_featurizer.blocks.{i}"

            # Get QKV weights and pad head_dim
            qkv_w = state_dict[f"{prefix}.attn.qkv.weight"]
            qkv_b = state_dict[f"{prefix}.attn.qkv.bias"]
            proj_w = state_dict[f"{prefix}.attn.proj.weight"]
            proj_b = state_dict[f"{prefix}.attn.proj.bias"]

            # Pad to head_dim=96
            qkv_w, qkv_b, proj_w, proj_b = upchannel_attn_weight_bias(qkv_w, qkv_b, proj_w, proj_b, num_heads=16)

            block_params = {
                "norm1_weight": ttnn.from_torch(
                    state_dict[f"{prefix}.norm1.weight"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "norm1_bias": ttnn.from_torch(
                    state_dict[f"{prefix}.norm1.bias"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "qkv_weight": ttnn.from_torch(
                    qkv_w.t().contiguous().to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "qkv_bias": ttnn.from_torch(
                    qkv_b.unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "proj_weight": ttnn.from_torch(
                    proj_w.t().contiguous().to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "proj_bias": ttnn.from_torch(
                    proj_b.unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "norm2_weight": ttnn.from_torch(
                    state_dict[f"{prefix}.norm2.weight"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "norm2_bias": ttnn.from_torch(
                    state_dict[f"{prefix}.norm2.bias"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "fc1_weight": ttnn.from_torch(
                    state_dict[f"{prefix}.mlp.fc1.weight"].t().contiguous().to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "fc1_bias": ttnn.from_torch(
                    state_dict[f"{prefix}.mlp.fc1.bias"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "fc2_weight": ttnn.from_torch(
                    state_dict[f"{prefix}.mlp.fc2.weight"].t().contiguous().to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
                "fc2_bias": ttnn.from_torch(
                    state_dict[f"{prefix}.mlp.fc2.bias"].unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
            }
            self.siglip_blocks.append(block_params)

        print("   Weight preprocessing complete!")

    def forward(self, pixel_values):
        """
        Forward pass.

        Args:
            pixel_values: [batch, 6, 224, 224] - first 3 channels for DINOv2, last 3 for SigLIP

        Returns:
            [batch, 256, 2176] - concatenated DINOv2 + SigLIP features
        """
        batch_size = pixel_values.shape[0]

        # Split input for DINOv2 and SigLIP
        # Input is NCHW, convert to NHWC and pad to 4 channels
        dinov2_in = pixel_values[:, :3, :, :]  # [B, 3, 224, 224]
        siglip_in = pixel_values[:, 3:, :, :]  # [B, 3, 224, 224]

        # Convert to TTNN format (NHWC + pad)
        dinov2_in = dinov2_in.permute(0, 2, 3, 1)
        dinov2_in = torch.nn.functional.pad(dinov2_in, (0, 1))
        dinov2_in = ttnn.from_torch(
            dinov2_in.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

        siglip_in = siglip_in.permute(0, 2, 3, 1)
        siglip_in = torch.nn.functional.pad(siglip_in, (0, 1))
        siglip_in = ttnn.from_torch(
            siglip_in.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

        # ========================
        # DINOv2 forward (all 24 layers)
        # ========================
        hidden_states = dinov2_patch_embeddings(
            dinov2_in,
            self.dinov2_patch_weight,
            self.dinov2_patch_bias,
            self.dinov2_pos_embed,
            self.dinov2_cls_token,
            self.dinov2_reg_token,
        )
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

        for block_params in self.dinov2_blocks:
            hidden_states = dinov2_block(hidden_states, block_params)

        # Final layer norm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.dinov2_final_norm_w,
            bias=self.dinov2_final_norm_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Skip CLS + 4 REG tokens
        dinov2_features = hidden_states[:, 5:, :]  # [B, 256, 1024]

        # ========================
        # SigLIP forward (all 27 layers)
        # ========================
        hidden_states = siglip_patch_embeddings(
            siglip_in,
            self.siglip_patch_weight,
            self.siglip_patch_bias,
            self.siglip_pos_embed,
        )
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

        for block_params in self.siglip_blocks:
            hidden_states = siglip_block(hidden_states, block_params)

        # Final layer norm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.siglip_final_norm_w,
            bias=self.siglip_final_norm_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        siglip_features = hidden_states  # [B, 256, 1152]

        # ========================
        # Concatenate (ensure same dtype)
        # ========================
        # Ensure both have same dtype for concat
        dinov2_features = ttnn.typecast(dinov2_features, dtype=ttnn.bfloat16)
        siglip_features = ttnn.typecast(siglip_features, dtype=ttnn.bfloat16)
        output = ttnn.concat([dinov2_features, siglip_features], dim=2)  # [B, 256, 2176]

        return output

    def __call__(self, pixel_values):
        return self.forward(pixel_values)
