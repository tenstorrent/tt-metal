# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ttnn_optimized_vit_smolvla.py

Optimized TT vision encoder for SmolVLA model.

SmolVLA uses a SigLIP-style vision encoder with:
- 12 transformer layers
- 768 hidden dimension
- 12 attention heads (64 head dim)
- 16×16 patch size
- 512×512 input images
- 1024 patches (32×32 grid)
- 3072 intermediate (MLP) dimension

This is different from OpenVLA's SigLIP SO400M which has:
- 27 layers, 1152 hidden, 16 heads, 14×14 patches, 224×224 images

References:
- SmolVLM2-500M: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- HuggingFace SiglipVisionModel
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn

# ============================================================================
# Configuration
# ============================================================================

# SmolVLA Vision Encoder Config
SMOLVLA_VISION_CONFIG = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "image_size": 512,
    "patch_size": 16,
    "layer_norm_eps": 1e-6,
}

# Core grid for TT operations - optimized for 1024 patches
# 1024 patches = 32x32, which divides well across 8x8 core grid
CORE_GRID = ttnn.CoreGrid(y=8, x=8)


# ============================================================================
# Patch Embeddings
# ============================================================================


def smolvla_patch_embeddings(
    pixel_values: ttnn.Tensor,
    proj_weight: ttnn.Tensor,
    proj_bias: ttnn.Tensor,
    patch_size: int = 16,
) -> ttnn.Tensor:
    """
    Convert image pixels to patch embeddings using optimized TT fold + linear.

    Args:
        pixel_values: [batch, H, W, 4] in NHWC format (padded from 3 to 4 channels)
        proj_weight: [patch_size*patch_size*4, hidden_size] preprocessed weight
        proj_bias: [1, hidden_size] bias

    Returns:
        [batch, num_patches, hidden_size] patch embeddings
    """
    batch_size, img_h, img_w, img_c = pixel_values.shape
    patch_count = img_h // patch_size  # 512 / 16 = 32
    patch_count_all = patch_count * patch_count  # 32 * 32 = 1024

    # Reshape for fold operation: [B, H, W/patch, 4*patch]
    pixel_values = ttnn.reshape(
        pixel_values,
        (batch_size, img_h, img_w // patch_size, img_c * patch_size),
    )

    # Fold patches: each patch becomes a row
    # stride_h = patch_size folds vertically, stride_w = 1 keeps horizontal
    pixel_values = ttnn.fold(pixel_values, patch_size, 1)

    # Convert to tile layout for efficient matmul
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    # Linear projection: [B*num_patches, patch_size*patch_size*4] -> [B*num_patches, hidden_size]
    patch_embeddings = ttnn.linear(
        pixel_values,
        proj_weight,
        bias=proj_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=CORE_GRID,
    )
    ttnn.deallocate(pixel_values)

    # Reshape to [batch, num_patches, hidden_size]
    patch_embeddings = ttnn.to_layout(patch_embeddings, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embeddings = ttnn.reshape(patch_embeddings, (batch_size, patch_count_all, -1))

    return patch_embeddings


# ============================================================================
# Self-Attention
# ============================================================================


def smolvla_attention(
    hidden_states: ttnn.Tensor,
    attention_mask: Optional[ttnn.Tensor],
    qkv_weight: ttnn.Tensor,
    qkv_bias: ttnn.Tensor,
    proj_weight: ttnn.Tensor,
    proj_bias: ttnn.Tensor,
    num_heads: int = 12,
) -> ttnn.Tensor:
    """
    Multi-head self-attention optimized for SmolVLA.

    Uses fused QKV projection and split_query_key_value_and_split_heads
    for efficient attention computation on TT hardware.

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        attention_mask: Optional attention mask
        qkv_weight: Fused Q/K/V projection weights [3*hidden_size, hidden_size]
        qkv_bias: Fused Q/K/V bias [3*hidden_size]
        proj_weight: Output projection weights [hidden_size, hidden_size]
        proj_bias: Output projection bias [hidden_size]
        num_heads: Number of attention heads (12 for SmolVLA)

    Returns:
        [batch, seq_len, hidden_size] attention output
    """
    *_, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads  # 768 / 12 = 64

    # Fused QKV projection
    query_key_value = ttnn.linear(
        hidden_states,
        qkv_weight,
        bias=qkv_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=CORE_GRID,
    )
    ttnn.reallocate(hidden_states)

    # Split into Q, K, V and reshape for multi-head attention
    # Output shapes: [batch, num_heads, seq_len, head_dim]
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)
    value = ttnn.reallocate(value)

    # Scale query by 1/sqrt(head_dim) before matmul
    scale = 1.0 / math.sqrt(head_dim)
    query = ttnn.mul(query, scale)

    # Scaled dot-product attention: (Q / sqrt(d)) @ K^T
    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=CORE_GRID,
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    # Softmax (in-place for efficiency) - no mask needed for vision
    attention_probs = ttnn.softmax_in_place(attention_scores, numeric_stable=True)

    # Attention output: softmax(Q @ K^T / sqrt(d)) @ V
    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=CORE_GRID,
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    # Concatenate heads: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Output projection
    attention_output = ttnn.linear(
        context_layer,
        proj_weight,
        bias=proj_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=CORE_GRID,
    )
    ttnn.deallocate(context_layer)

    return attention_output


# ============================================================================
# MLP / Feed-Forward
# ============================================================================


def smolvla_mlp(
    hidden_states: ttnn.Tensor,
    fc1_weight: ttnn.Tensor,
    fc1_bias: ttnn.Tensor,
    fc2_weight: ttnn.Tensor,
    fc2_bias: ttnn.Tensor,
) -> ttnn.Tensor:
    """
    MLP block with GELU activation.

    Architecture: hidden_size -> intermediate_size (GELU) -> hidden_size
    For SmolVLA: 768 -> 3072 -> 768

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        fc1_weight: [intermediate_size, hidden_size]
        fc1_bias: [intermediate_size]
        fc2_weight: [hidden_size, intermediate_size]
        fc2_bias: [hidden_size]

    Returns:
        [batch, seq_len, hidden_size] MLP output
    """
    # Up projection with GELU
    intermediate = ttnn.linear(
        hidden_states,
        fc1_weight,
        bias=fc1_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=CORE_GRID,
        activation="gelu",
    )

    # Down projection
    output = ttnn.linear(
        intermediate,
        fc2_weight,
        bias=fc2_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        core_grid=CORE_GRID,
    )
    ttnn.deallocate(intermediate)

    return output


# ============================================================================
# Transformer Layer
# ============================================================================


def smolvla_encoder_layer(
    hidden_states: ttnn.Tensor,
    attention_mask: Optional[ttnn.Tensor],
    layer_params: Dict[str, ttnn.Tensor],
    layer_norm_eps: float = 1e-6,
) -> ttnn.Tensor:
    """
    Single transformer encoder layer with pre-norm architecture.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        attention_mask: Optional attention mask
        layer_params: Dictionary containing all layer weights
        layer_norm_eps: LayerNorm epsilon

    Returns:
        [batch, seq_len, hidden_size] layer output
    """
    # Pre-attention LayerNorm
    normed_hidden = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["norm1_weight"],
        bias=layer_params["norm1_bias"],
        epsilon=layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Self-attention
    attention_output = smolvla_attention(
        normed_hidden,
        attention_mask,
        layer_params["qkv_weight"],
        layer_params["qkv_bias"],
        layer_params["proj_weight"],
        layer_params["proj_bias"],
        num_heads=12,
    )

    # Residual connection
    hidden_states = ttnn.add(
        attention_output,
        hidden_states,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(attention_output)

    # Pre-MLP LayerNorm
    normed_hidden = ttnn.layer_norm(
        hidden_states,
        weight=layer_params["norm2_weight"],
        bias=layer_params["norm2_bias"],
        epsilon=layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # MLP
    mlp_output = smolvla_mlp(
        normed_hidden,
        layer_params["fc1_weight"],
        layer_params["fc1_bias"],
        layer_params["fc2_weight"],
        layer_params["fc2_bias"],
    )

    # Residual connection
    hidden_states = ttnn.add(
        mlp_output,
        hidden_states,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(mlp_output)

    return hidden_states


# ============================================================================
# Full Encoder
# ============================================================================


def smolvla_encoder(
    embeddings: ttnn.Tensor,
    attention_masks: List[Optional[ttnn.Tensor]],
    layer_params_list: List[Dict[str, ttnn.Tensor]],
    layer_norm_eps: float = 1e-6,
) -> ttnn.Tensor:
    """
    Full 12-layer transformer encoder for SmolVLA vision.

    Args:
        embeddings: [batch, num_patches, hidden_size] patch embeddings with position
        attention_masks: List of attention masks per layer (can be None)
        layer_params_list: List of parameter dicts for each layer

    Returns:
        [batch, num_patches, hidden_size] encoded features
    """
    hidden_states = embeddings

    for layer_idx, layer_params in enumerate(layer_params_list):
        mask = attention_masks[layer_idx] if attention_masks else None
        hidden_states = smolvla_encoder_layer(
            hidden_states,
            mask,
            layer_params,
            layer_norm_eps,
        )

    return hidden_states


# ============================================================================
# Full Vision Model Forward
# ============================================================================


def smolvla_vision_forward(
    pixel_values: ttnn.Tensor,
    position_embeddings: ttnn.Tensor,
    patch_proj_weight: ttnn.Tensor,
    patch_proj_bias: ttnn.Tensor,
    layer_params_list: List[Dict[str, ttnn.Tensor]],
    post_layernorm_weight: ttnn.Tensor,
    post_layernorm_bias: ttnn.Tensor,
    layer_norm_eps: float = 1e-6,
) -> ttnn.Tensor:
    """
    Complete SmolVLA vision encoder forward pass.

    Pipeline:
    1. Patch embedding (fold + linear)
    2. Add position embeddings
    3. 12 transformer layers
    4. Post-LayerNorm

    Args:
        pixel_values: [batch, H, W, 4] in NHWC format (padded channels)
        position_embeddings: [1, num_patches, hidden_size]
        patch_proj_weight: Preprocessed patch projection weight
        patch_proj_bias: Patch projection bias
        layer_params_list: List of per-layer parameters
        post_layernorm_weight: Final LayerNorm weight
        post_layernorm_bias: Final LayerNorm bias

    Returns:
        [batch, num_patches, hidden_size] vision features
    """
    # Patch embeddings
    patch_embeddings = smolvla_patch_embeddings(
        pixel_values,
        patch_proj_weight,
        patch_proj_bias,
        patch_size=16,
    )

    # Add position embeddings
    embeddings = ttnn.add(
        patch_embeddings,
        position_embeddings,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(patch_embeddings)

    # Encoder layers (no attention masks needed for vision)
    attention_masks = [None] * len(layer_params_list)
    hidden_states = smolvla_encoder(
        embeddings,
        attention_masks,
        layer_params_list,
        layer_norm_eps,
    )

    # Post-LayerNorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=post_layernorm_weight,
        bias=post_layernorm_bias,
        epsilon=layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return output


# ============================================================================
# Weight Preprocessing
# ============================================================================


def preprocess_patch_embedding_weights(
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: Any,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Preprocess patch embedding conv weights for TT linear.

    Converts [out_channels, in_channels, kH, kW] conv weight to
    [kH*kW*in_channels_padded, out_channels] linear weight.

    Args:
        weight: [768, 3, 16, 16] conv weight
        bias: [768] bias
        device: TT device

    Returns:
        (proj_weight, proj_bias) preprocessed for TT
    """
    out_channels, in_channels, kh, kw = weight.shape

    # Pad input channels from 3 to 4
    pad_value = 4 - in_channels
    weight_padded = F.pad(weight, (0, 0, 0, 0, 0, pad_value))  # [768, 4, 16, 16]

    # Reshape for linear: [kH, kW, in_c_padded, out_c] -> [kH*kW*in_c_padded, out_c]
    weight_reshaped = weight_padded.permute(2, 3, 1, 0)  # [16, 16, 4, 768]
    weight_reshaped = weight_reshaped.reshape(-1, out_channels)  # [1024, 768]

    proj_weight = ttnn.from_torch(
        weight_reshaped.to(torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    proj_bias = ttnn.from_torch(
        bias.unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    return proj_weight, proj_bias


def preprocess_attention_weights(
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    device: Any,
) -> Dict[str, ttnn.Tensor]:
    """
    Preprocess separate Q/K/V weights into fused QKV format for TT.

    Args:
        q/k/v_weight: [hidden_size, hidden_size] projection weights
        q/k/v_bias: [hidden_size] biases
        out_weight: [hidden_size, hidden_size] output projection
        out_bias: [hidden_size] output bias
        device: TT device

    Returns:
        Dict with qkv_weight, qkv_bias, proj_weight, proj_bias
    """
    # Fuse Q, K, V weights
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # [3*hidden, hidden]
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)  # [3*hidden]

    # Preprocess and transfer to device
    qkv_w = preprocess_linear_weight(qkv_weight.to(torch.bfloat16), dtype=ttnn.bfloat8_b)
    qkv_b = preprocess_linear_bias(qkv_bias.to(torch.bfloat16), dtype=ttnn.bfloat8_b)
    proj_w = preprocess_linear_weight(out_weight.to(torch.bfloat16), dtype=ttnn.bfloat8_b)
    proj_b = preprocess_linear_bias(out_bias.to(torch.bfloat16), dtype=ttnn.bfloat8_b)

    return {
        "qkv_weight": ttnn.to_device(qkv_w, device),
        "qkv_bias": ttnn.to_device(qkv_b, device),
        "proj_weight": ttnn.to_device(proj_w, device),
        "proj_bias": ttnn.to_device(proj_b, device),
    }


def preprocess_mlp_weights(
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    device: Any,
) -> Dict[str, ttnn.Tensor]:
    """
    Preprocess MLP weights for TT.

    Args:
        fc1_weight: [intermediate_size, hidden_size]
        fc1_bias: [intermediate_size]
        fc2_weight: [hidden_size, intermediate_size]
        fc2_bias: [hidden_size]
        device: TT device

    Returns:
        Dict with fc1_weight, fc1_bias, fc2_weight, fc2_bias
    """
    # Preprocess and transfer to device
    fc1_w = preprocess_linear_weight(fc1_weight.to(torch.bfloat16), dtype=ttnn.bfloat8_b)
    fc1_b = preprocess_linear_bias(fc1_bias.to(torch.bfloat16), dtype=ttnn.bfloat8_b)
    fc2_w = preprocess_linear_weight(fc2_weight.to(torch.bfloat16), dtype=ttnn.bfloat8_b)
    fc2_b = preprocess_linear_bias(fc2_bias.to(torch.bfloat16), dtype=ttnn.bfloat8_b)

    return {
        "fc1_weight": ttnn.to_device(fc1_w, device),
        "fc1_bias": ttnn.to_device(fc1_b, device),
        "fc2_weight": ttnn.to_device(fc2_w, device),
        "fc2_bias": ttnn.to_device(fc2_b, device),
    }


def preprocess_layernorm_weights(
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: Any,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Preprocess LayerNorm weights for TT.

    Args:
        weight: [hidden_size] LayerNorm weight
        bias: [hidden_size] LayerNorm bias
        device: TT device

    Returns:
        (weight_tt, bias_tt)
    """
    weight_tt = ttnn.from_torch(
        weight.unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    bias_tt = ttnn.from_torch(
        bias.unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return weight_tt, bias_tt


# ============================================================================
# SmolVLA Vision Encoder Class
# ============================================================================


class SmolVLAVisionEncoderOptimized:
    """
    Optimized TT implementation of SmolVLA vision encoder.

    This class handles weight preprocessing and provides an efficient
    forward pass using the optimized TT operations defined above.
    """

    def __init__(
        self,
        vision_model: torch.nn.Module,
        device: Any,
        config: Optional[Dict] = None,
    ):
        """
        Initialize from a HuggingFace SiglipVisionModel.

        Args:
            vision_model: HF SiglipVisionModel instance with loaded weights
            device: TT device
            config: Optional config dict (uses defaults if not provided)
        """
        self.device = device
        self.config = config or SMOLVLA_VISION_CONFIG

        # Preprocess all weights
        self._preprocess_weights(vision_model)

    def _preprocess_weights(self, vision_model: torch.nn.Module):
        """Extract and preprocess all weights from HF model."""
        vm = vision_model.vision_model if hasattr(vision_model, "vision_model") else vision_model

        # Patch embedding
        patch_embed = vm.embeddings.patch_embedding
        self.patch_proj_weight, self.patch_proj_bias = preprocess_patch_embedding_weights(
            patch_embed.weight,
            patch_embed.bias if patch_embed.bias is not None else torch.zeros(self.config["hidden_size"]),
            self.device,
        )

        # Position embeddings
        pos_embed = vm.embeddings.position_embedding.weight  # [num_patches, hidden_size]
        self.position_embeddings = ttnn.from_torch(
            pos_embed.unsqueeze(0).to(torch.bfloat16),  # [1, 1024, 768]
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Encoder layers
        self.layer_params_list = []
        for layer in vm.encoder.layers:
            layer_params = {}

            # LayerNorm 1 (pre-attention)
            layer_params["norm1_weight"], layer_params["norm1_bias"] = preprocess_layernorm_weights(
                layer.layer_norm1.weight,
                layer.layer_norm1.bias,
                self.device,
            )

            # Attention
            attn = layer.self_attn
            attn_params = preprocess_attention_weights(
                attn.q_proj.weight,
                attn.q_proj.bias,
                attn.k_proj.weight,
                attn.k_proj.bias,
                attn.v_proj.weight,
                attn.v_proj.bias,
                attn.out_proj.weight,
                attn.out_proj.bias,
                self.device,
            )
            layer_params.update(attn_params)

            # LayerNorm 2 (pre-MLP)
            layer_params["norm2_weight"], layer_params["norm2_bias"] = preprocess_layernorm_weights(
                layer.layer_norm2.weight,
                layer.layer_norm2.bias,
                self.device,
            )

            # MLP
            mlp = layer.mlp
            mlp_params = preprocess_mlp_weights(
                mlp.fc1.weight,
                mlp.fc1.bias,
                mlp.fc2.weight,
                mlp.fc2.bias,
                self.device,
            )
            layer_params.update(mlp_params)

            self.layer_params_list.append(layer_params)

        # Post-LayerNorm
        self.post_ln_weight, self.post_ln_bias = preprocess_layernorm_weights(
            vm.post_layernorm.weight,
            vm.post_layernorm.bias,
            self.device,
        )

    def __call__(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        Forward pass through the vision encoder.

        Args:
            pixel_values: [batch, channels, height, width] in NCHW format

        Returns:
            [batch, num_patches, hidden_size] vision features as TT tensor
        """
        batch_size = pixel_values.shape[0]

        # Convert NCHW -> NHWC and pad channels 3 -> 4
        if pixel_values.shape[1] == 3:
            pixel_values_nhwc = pixel_values.permute(0, 2, 3, 1)  # [B, H, W, 3]
            pixel_values_nhwc = F.pad(pixel_values_nhwc, (0, 1))  # [B, H, W, 4]
        else:
            pixel_values_nhwc = pixel_values.permute(0, 2, 3, 1)

        # Transfer to TT
        pixel_values_tt = ttnn.from_torch(
            pixel_values_nhwc.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        # Forward through vision encoder
        output = smolvla_vision_forward(
            pixel_values_tt,
            self.position_embeddings,
            self.patch_proj_weight,
            self.patch_proj_bias,
            self.layer_params_list,
            self.post_ln_weight,
            self.post_ln_bias,
            self.config.get("layer_norm_eps", 1e-6),
        )

        return output


# ============================================================================
# Factory Function
# ============================================================================


def create_smolvla_vision_encoder(
    vision_model: torch.nn.Module,
    device: Any,
) -> SmolVLAVisionEncoderOptimized:
    """
    Create an optimized SmolVLA vision encoder from a HuggingFace model.

    Args:
        vision_model: HF SiglipVisionModel with loaded weights
        device: TT device

    Returns:
        SmolVLAVisionEncoderOptimized instance ready for inference
    """
    return SmolVLAVisionEncoderOptimized(vision_model, device)
