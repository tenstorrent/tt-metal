# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
SigLIP2 Vision Encoder - TTNN Implementation for GR00T N1.6.

SigLIP2 architecture (Eagle-Block2A-2B-v2):
    - 27 transformer layers
    - 1152 hidden dimension
    - 16 attention heads (72 head dim)
    - 14x14 patch size
    - Learned position embeddings
    - Post-LayerNorm

Adapted from the pi0 SigLIP implementation with adjustments for SigLIP2 dimensions.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.groot_n16.common.configs import SigLIP2Config
from models.experimental.groot_n16.tt.ttnn_common import (
    CORE_GRID_BH,
    nearest_32,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_params,
)


def siglip2_patch_embeddings_cpu(
    pixel_values: torch.Tensor,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Extract patches from image on CPU.

    Args:
        pixel_values: [batch, 3, H, W] in NCHW format

    Returns:
        [batch, num_patches, patch_size*patch_size*3] flattened patches
    """
    batch_size, channels, h, w = pixel_values.shape
    patch_h = h // patch_size
    patch_w = w // patch_size

    # Reshape to extract patches: [B, C, pH, ps, pW, ps]
    x = pixel_values.reshape(batch_size, channels, patch_h, patch_size, patch_w, patch_size)
    # Permute to [B, pH, pW, ps, ps, C] then flatten patches
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.reshape(batch_size, patch_h * patch_w, patch_size * patch_size * channels)
    return x


def siglip2_attention(
    hidden_states: ttnn.Tensor,
    qkv_weight: ttnn.Tensor,
    qkv_bias: ttnn.Tensor,
    proj_weight: ttnn.Tensor,
    proj_bias: ttnn.Tensor,
    num_heads: int = 16,
) -> ttnn.Tensor:
    """
    Multi-head self-attention for SigLIP2.

    Uses fused QKV projection with nlp_create_qkv_heads for head_dim=72
    (which is not tile-aligned). The op handles padding internally.
    Uses SDPA for efficient attention computation.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape

    # Reshape to 4D for nlp_create_qkv_heads: [B, 1, seq, hidden]
    hidden_4d = ttnn.reshape(hidden_states, (batch_size, 1, seq_len, hidden_size))

    # Fused QKV projection
    qkv = ttnn.linear(
        hidden_4d, qkv_weight, bias=qkv_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
        core_grid=CORE_GRID_BH,
    )

    # Split into Q, K, V heads with padding handled internally
    q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
        qkv,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        transpose_k_heads=False,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(qkv)

    # Use scaled_dot_product_attention
    attn_output = ttnn.transformer.scaled_dot_product_attention(
        q_heads, k_heads, v_heads,
        is_causal=False,
    )
    ttnn.deallocate(q_heads)
    ttnn.deallocate(k_heads)
    ttnn.deallocate(v_heads)

    # Concatenate heads back
    attn_concat = ttnn.experimental.nlp_concat_heads(
        attn_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(attn_output)

    # Reshape back to 3D: [B, 1, seq, padded_hidden] -> [B, seq, hidden]
    # Slice to remove padding from head_dim (padded 72->96 = 96*16=1536 vs 72*16=1152)
    attn_3d = ttnn.reshape(attn_concat, (batch_size, seq_len, -1))
    ttnn.deallocate(attn_concat)

    # If padded, slice back to hidden_size
    if attn_3d.shape[-1] > hidden_size:
        attn_3d = ttnn.slice(attn_3d, [0, 0, 0], [batch_size, seq_len, hidden_size])

    # Output projection
    output = ttnn.linear(
        attn_3d, proj_weight, bias=proj_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b,
        core_grid=CORE_GRID_BH,
    )
    ttnn.deallocate(attn_3d)

    return output


def siglip2_mlp(
    hidden_states: ttnn.Tensor,
    fc1_weight: ttnn.Tensor,
    fc1_bias: ttnn.Tensor,
    fc2_weight: ttnn.Tensor,
    fc2_bias: ttnn.Tensor,
) -> ttnn.Tensor:
    """MLP with GELU activation: hidden -> intermediate -> hidden."""
    intermediate = ttnn.linear(
        hidden_states, fc1_weight, bias=fc1_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=CORE_GRID_BH,
        activation="gelu",
    )

    output = ttnn.linear(
        intermediate, fc2_weight, bias=fc2_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=CORE_GRID_BH,
    )
    ttnn.deallocate(intermediate)
    return output


def siglip2_encoder_layer(
    hidden_states: ttnn.Tensor,
    layer_params: Dict[str, ttnn.Tensor],
    layer_norm_eps: float = 1e-6,
) -> ttnn.Tensor:
    """Single SigLIP2 transformer layer with pre-norm."""
    # Pre-attention LayerNorm
    normed = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["norm1_weight"],
        bias=layer_params["norm1_bias"],
        epsilon=layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    attn_out = siglip2_attention(
        normed,
        layer_params["qkv_weight"],
        layer_params["qkv_bias"],
        layer_params["proj_weight"],
        layer_params["proj_bias"],
        num_heads=16,
    )

    hidden_states = ttnn.add(
        attn_out, hidden_states,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(attn_out)

    # Pre-MLP LayerNorm
    normed = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["norm2_weight"],
        bias=layer_params["norm2_bias"],
        epsilon=layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    mlp_out = siglip2_mlp(
        normed,
        layer_params["fc1_weight"],
        layer_params["fc1_bias"],
        layer_params["fc2_weight"],
        layer_params["fc2_bias"],
    )

    hidden_states = ttnn.add(
        mlp_out, hidden_states,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(mlp_out)

    return hidden_states


class SigLIP2VisionEncoderTTNN:
    """
    Complete SigLIP2 vision encoder for GR00T N1.6 on TTNN.

    27 layers, 1152 hidden, 16 heads, 14x14 patches.
    """

    def __init__(
        self,
        config: SigLIP2Config,
        weights: Dict[str, torch.Tensor],
        device: Any,
    ):
        self.config = config
        self.device = device
        self._preprocess_weights(weights)

    def _preprocess_weights(self, weights: Dict[str, torch.Tensor]):
        """Preprocess all SigLIP2 weights for TTNN."""
        # Patch embedding: conv2d -> linear
        # Patch embedding weight is already in linear format: [1152, 588]
        # where 588 = 3 * 14 * 14 (in_channels * patch_h * patch_w)
        linear_weight = weights.get("embeddings.patch_embedding.weight")
        linear_bias = weights.get("embeddings.patch_embedding.bias")

        if linear_weight is not None:
            # Weight is [out_features, in_features] = [1152, 588]
            # preprocess_linear_weight transposes to [588, 1152] for ttnn
            # TTNN handles tile padding internally
            self.patch_proj_weight = preprocess_linear_weight(linear_weight, self.device)
            self.patch_proj_bias = preprocess_linear_bias(
                linear_bias, self.device,
            ) if linear_bias is not None else None

        # Position embeddings
        pos_embed = weights.get("embeddings.position_embedding.weight")
        if pos_embed is not None:
            self.position_embeddings = ttnn.from_torch(
                pos_embed.unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        # Encoder layers
        self.layer_params_list = []
        for i in range(self.config.num_hidden_layers):
            prefix = f"encoder.layers.{i}."
            layer_params = {}

            # LayerNorm 1
            ln1_w = weights.get(f"{prefix}layer_norm1.weight")
            ln1_b = weights.get(f"{prefix}layer_norm1.bias")
            if ln1_w is not None:
                layer_params["norm1_weight"], layer_params["norm1_bias"] = \
                    preprocess_layernorm_params(ln1_w, ln1_b, self.device)

            # Fused QKV attention weights
            attn_prefix = f"{prefix}self_attn."
            q_w = weights.get(f"{attn_prefix}q_proj.weight")
            k_w = weights.get(f"{attn_prefix}k_proj.weight")
            v_w = weights.get(f"{attn_prefix}v_proj.weight")
            q_b = weights.get(f"{attn_prefix}q_proj.bias")
            k_b = weights.get(f"{attn_prefix}k_proj.bias")
            v_b = weights.get(f"{attn_prefix}v_proj.bias")

            if q_w is not None:
                qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
                qkv_b = torch.cat([q_b, k_b, v_b], dim=0)
                layer_params["qkv_weight"] = preprocess_linear_weight(qkv_w, self.device)
                layer_params["qkv_bias"] = preprocess_linear_bias(qkv_b, self.device)

            out_w = weights.get(f"{attn_prefix}out_proj.weight")
            out_b = weights.get(f"{attn_prefix}out_proj.bias")
            if out_w is not None:
                layer_params["proj_weight"] = preprocess_linear_weight(out_w, self.device)
                layer_params["proj_bias"] = preprocess_linear_bias(out_b, self.device)

            # LayerNorm 2
            ln2_w = weights.get(f"{prefix}layer_norm2.weight")
            ln2_b = weights.get(f"{prefix}layer_norm2.bias")
            if ln2_w is not None:
                layer_params["norm2_weight"], layer_params["norm2_bias"] = \
                    preprocess_layernorm_params(ln2_w, ln2_b, self.device)

            # MLP
            fc1_w = weights.get(f"{prefix}mlp.fc1.weight")
            fc1_b = weights.get(f"{prefix}mlp.fc1.bias")
            fc2_w = weights.get(f"{prefix}mlp.fc2.weight")
            fc2_b = weights.get(f"{prefix}mlp.fc2.bias")
            if fc1_w is not None:
                layer_params["fc1_weight"] = preprocess_linear_weight(fc1_w, self.device)
                layer_params["fc1_bias"] = preprocess_linear_bias(fc1_b, self.device)
                layer_params["fc2_weight"] = preprocess_linear_weight(fc2_w, self.device)
                layer_params["fc2_bias"] = preprocess_linear_bias(fc2_b, self.device)

            self.layer_params_list.append(layer_params)

        # Post-LayerNorm
        post_ln_w = weights.get("post_layernorm.weight")
        post_ln_b = weights.get("post_layernorm.bias")
        if post_ln_w is not None:
            self.post_ln_weight, self.post_ln_bias = \
                preprocess_layernorm_params(post_ln_w, post_ln_b, self.device)

    def __call__(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        Forward pass through SigLIP2 vision encoder.

        Args:
            pixel_values: [batch, 3, H, W] in NCHW format

        Returns:
            [batch, num_patches, hidden_size] vision features
        """
        # Extract patches on CPU: [B, num_patches, 588]
        patches = siglip2_patch_embeddings_cpu(
            pixel_values, patch_size=self.config.patch_size,
        )

        # Transfer to device and project
        patches_tt = ttnn.from_torch(
            patches.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Linear projection: [B, num_patches, 588] -> [B, num_patches, 1152]
        embeddings = ttnn.linear(
            patches_tt, self.patch_proj_weight, bias=self.patch_proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(patches_tt)

        # Add position embeddings
        embeddings = ttnn.add(
            embeddings, self.position_embeddings,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        # Encoder layers
        hidden_states = embeddings
        for layer_params in self.layer_params_list:
            hidden_states = siglip2_encoder_layer(
                hidden_states, layer_params, self.config.layer_norm_eps,
            )

        # Post-LayerNorm
        output = ttnn.layer_norm(
            hidden_states,
            weight=self.post_ln_weight,
            bias=self.post_ln_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return output
