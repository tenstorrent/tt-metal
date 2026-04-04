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
from typing import Any, Dict

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.gr00t_n1_6.common.configs import SigLIP2Config
from models.experimental.gr00t_n1_6.tt.ttnn_common import (
    CORE_GRID_BH,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_params,
)


def siglip2_patch_embeddings_cpu(
    pixel_values: torch.Tensor,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Extract patches from image on CPU in (C, H, W) order per patch.

    The weight is stored as [out_features, C*pH*pW] in (c,h,w) order,
    so patch features must also be flattened in (c,h,w) order.

    Args:
        pixel_values: [batch, 3, H, W] in NCHW format

    Returns:
        [batch, num_patches, C*patch_size*patch_size] flattened patches in (c,h,w) order
    """
    batch_size, channels, h, w = pixel_values.shape
    patch_h = h // patch_size
    patch_w = w // patch_size

    # Reshape: [B, C, pH, ps, pW, ps]
    x = pixel_values.reshape(batch_size, channels, patch_h, patch_size, patch_w, patch_size)
    # Permute to [B, pH, pW, C, ps, ps] — keeps (C, h, w) order within each patch
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    # Flatten: [B, num_patches, C*ps*ps]
    x = x.reshape(batch_size, patch_h * patch_w, channels * patch_size * patch_size)
    return x


SIGLIP2_PADDED_HEAD_DIM = 96  # 72 -> 96 (nearest multiple of 32)


def _pad_qkv_weight_siglip2(
    qkv_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Pad fused QKV weight: [3*num_heads*head_dim, hidden] -> [3*num_heads*padded_hd, hidden]."""
    out_dim, in_dim = qkv_weight.shape
    # Split into Q, K, V sections
    per_section = num_heads * head_dim
    sections = []
    for i in range(3):
        w = qkv_weight[i * per_section : (i + 1) * per_section]  # [num_heads*head_dim, hidden]
        w = w.reshape(num_heads, head_dim, in_dim)
        w = F.pad(w, (0, 0, 0, padded_head_dim - head_dim))  # pad head_dim
        w = w.reshape(num_heads * padded_head_dim, in_dim)
        sections.append(w)
    return torch.cat(sections, dim=0)  # [3*num_heads*padded_hd, hidden]


def _pad_qkv_bias_siglip2(
    qkv_bias: torch.Tensor,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Pad fused QKV bias: [3*num_heads*head_dim] -> [3*num_heads*padded_hd]."""
    per_section = num_heads * head_dim
    sections = []
    for i in range(3):
        b = qkv_bias[i * per_section : (i + 1) * per_section].reshape(num_heads, head_dim)
        b = F.pad(b, (0, padded_head_dim - head_dim))
        sections.append(b.reshape(num_heads * padded_head_dim))
    return torch.cat(sections, dim=0)


def _pad_out_proj_weight_siglip2(
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Pad output projection weight input dim: [hidden, num_heads*head_dim] -> [hidden, num_heads*padded_hd]."""
    out_dim, in_dim = weight.shape
    w = weight.reshape(out_dim, num_heads, head_dim)
    w = F.pad(w, (0, padded_head_dim - head_dim))
    return w.reshape(out_dim, num_heads * padded_head_dim)


def siglip2_attention(
    hidden_states: ttnn.Tensor,
    qkv_weight: ttnn.Tensor,
    qkv_bias: ttnn.Tensor,
    proj_weight: ttnn.Tensor,
    proj_bias: ttnn.Tensor,
    num_heads: int = 16,
    head_dim: int = 72,
    padded_head_dim: int = SIGLIP2_PADDED_HEAD_DIM,
) -> ttnn.Tensor:
    """
    Fully on-device multi-head self-attention for SigLIP2.

    Weights are pre-padded from head_dim=72 to padded_head_dim=96 (tile-aligned).
    Uses reshape+permute+matmul entirely on device.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape

    # Fused QKV projection: [B, seq, hidden] -> [B, seq, 3*num_heads*padded_hd]
    qkv = ttnn.linear(
        hidden_states,
        qkv_weight,
        bias=qkv_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID_BH,
    )

    # Split Q, K, V and reshape to multi-head
    padded_dim = num_heads * padded_head_dim  # 16 * 96 = 1536
    q = ttnn.slice(qkv, [0, 0, 0], [batch_size, seq_len, padded_dim])
    k = ttnn.slice(qkv, [0, 0, padded_dim], [batch_size, seq_len, 2 * padded_dim])
    v = ttnn.slice(qkv, [0, 0, 2 * padded_dim], [batch_size, seq_len, 3 * padded_dim])
    ttnn.deallocate(qkv)

    # Reshape: [B, seq, heads*padded_hd] -> [B, heads, seq, padded_hd]
    q = ttnn.reshape(q, (batch_size, seq_len, num_heads, padded_head_dim))
    q = ttnn.permute(q, (0, 2, 1, 3))
    k = ttnn.reshape(k, (batch_size, seq_len, num_heads, padded_head_dim))
    k = ttnn.permute(k, (0, 2, 1, 3))
    v = ttnn.reshape(v, (batch_size, seq_len, num_heads, padded_head_dim))
    v = ttnn.permute(v, (0, 2, 1, 3))

    # Scaled dot-product: Q @ K^T / sqrt(head_dim)
    scale = 1.0 / math.sqrt(head_dim)  # Use REAL head_dim=72, not padded
    q = ttnn.mul(q, scale)
    k = ttnn.permute(k, (0, 1, 3, 2))  # Transpose K

    attn = ttnn.matmul(q, k, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
    ttnn.deallocate(q)
    ttnn.deallocate(k)

    attn = ttnn.softmax_in_place(attn, numeric_stable=True)

    context = ttnn.matmul(attn, v, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
    ttnn.deallocate(attn)
    ttnn.deallocate(v)

    # Reshape back: [B, heads, seq, padded_hd] -> [B, seq, heads*padded_hd]
    context = ttnn.permute(context, (0, 2, 1, 3))
    context = ttnn.reshape(context, (batch_size, seq_len, num_heads * padded_head_dim))

    # Output projection (weight already padded for input)
    output = ttnn.linear(
        context,
        proj_weight,
        bias=proj_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID_BH,
    )
    ttnn.deallocate(context)

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
        hidden_states,
        fc1_weight,
        bias=fc1_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID_BH,
        activation="gelu",
    )

    output = ttnn.linear(
        intermediate,
        fc2_weight,
        bias=fc2_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
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
        attn_out,
        hidden_states,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
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
        mlp_out,
        hidden_states,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
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
            self.patch_proj_bias = (
                preprocess_linear_bias(
                    linear_bias,
                    self.device,
                )
                if linear_bias is not None
                else None
            )

        # Position embeddings
        pos_embed = weights.get("embeddings.position_embedding.weight")
        if pos_embed is not None:
            self.position_embeddings = ttnn.from_torch(
                pos_embed.unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
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
                layer_params["norm1_weight"], layer_params["norm1_bias"] = preprocess_layernorm_params(
                    ln1_w, ln1_b, self.device
                )

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
                # Pad QKV weights for tile-aligned head_dim (72 -> 96)
                num_heads = self.config.num_attention_heads
                head_dim = self.config.head_dim
                qkv_w_padded = _pad_qkv_weight_siglip2(qkv_w, num_heads, head_dim, SIGLIP2_PADDED_HEAD_DIM)
                qkv_b_padded = _pad_qkv_bias_siglip2(qkv_b, num_heads, head_dim, SIGLIP2_PADDED_HEAD_DIM)
                layer_params["qkv_weight"] = preprocess_linear_weight(qkv_w_padded, self.device)
                layer_params["qkv_bias"] = preprocess_linear_bias(qkv_b_padded, self.device)

            out_w = weights.get(f"{attn_prefix}out_proj.weight")
            out_b = weights.get(f"{attn_prefix}out_proj.bias")
            if out_w is not None:
                # Pad output projection input dim for padded heads
                out_w_padded = _pad_out_proj_weight_siglip2(out_w, num_heads, head_dim, SIGLIP2_PADDED_HEAD_DIM)
                layer_params["proj_weight"] = preprocess_linear_weight(out_w_padded, self.device)
                layer_params["proj_bias"] = preprocess_linear_bias(out_b, self.device)

            # LayerNorm 2
            ln2_w = weights.get(f"{prefix}layer_norm2.weight")
            ln2_b = weights.get(f"{prefix}layer_norm2.bias")
            if ln2_w is not None:
                layer_params["norm2_weight"], layer_params["norm2_bias"] = preprocess_layernorm_params(
                    ln2_w, ln2_b, self.device
                )

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
            self.post_ln_weight, self.post_ln_bias = preprocess_layernorm_params(post_ln_w, post_ln_b, self.device)

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
            pixel_values,
            patch_size=self.config.patch_size,
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
            patches_tt,
            self.patch_proj_weight,
            bias=self.patch_proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(patches_tt)

        # Add position embeddings
        embeddings = ttnn.add(
            embeddings,
            self.position_embeddings,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        # Encoder layers
        hidden_states = embeddings
        for layer_params in self.layer_params_list:
            hidden_states = siglip2_encoder_layer(
                hidden_states,
                layer_params,
                self.config.layer_norm_eps,
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
