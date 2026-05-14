# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Implementation of Depth-Anything-V2-Large.

This module implements the full Depth-Anything-V2-Large model using TTNN APIs
for inference on Tenstorrent Wormhole hardware.

Architecture Overview:
  1. Patch Embedding: Conv2d(3, 1024, kernel_size=14, stride=14) + CLS token + positional encoding
  2. ViT-L/14 Backbone: 24 transformer blocks (LayerNorm -> Attention -> LayerNorm -> MLP)
     with LayerScale (init_values=1.0)
  3. DPT Head: Extract features from 4 intermediate layers, process through
     projection, resize, fusion blocks, and output convolutions

Key differences from standard ViT:
  - DINOv2-style attention (MemEffAttention with qkv_bias, separate proj_bias, ffn_bias)
  - LayerScale: multiply attention and mlp outputs by learnable scale (init=1.0)
  - Intermediate feature extraction at layers [4, 11, 17, 23]
  - No class token used in DPT head (use_clstoken=False for Large variant)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn

from .depth_anything_v2_config import DepthAnythingV2Config


# =============================================================================
# Utility Functions
# =============================================================================


def pad_to_tile_dim(dim: int, tile_size: int = 32) -> int:
    """Pad dimension to be divisible by tile size."""
    return ((dim + tile_size - 1) // tile_size) * tile_size


def pre_process_input(x: torch.Tensor, config: DepthAnythingV2Config) -> torch.Tensor:
    """Preprocess input image for TTNN inference.

    Args:
        x: Input image tensor [B, 3, H, W] normalized with ImageNet stats
        config: Model configuration

    Returns:
        Preprocessed tensor ready for patch embedding
    """
    B, C, H, W = x.shape
    assert H == config.image_size and W == config.image_size, (
        f"Expected {config.image_size}x{config.image_size}, got {H}x{W}"
    )
    return x


# =============================================================================
# Weight Preprocessing
# =============================================================================


def preprocess_patch_embed_weights(
    model, device: ttnn.Device, config: DepthAnythingV2Config
) -> Dict[str, ttnn.Tensor]:
    """Preprocess patch embedding weights for TTNN.

    The patch embedding is a Conv2d(3, 1024, kernel_size=14, stride=14) with no bias.
    """
    state_dict = model.state_dict()

    parameters = {}

    # Patch embedding conv weight: [1024, 3, 14, 14]
    patch_embed_weight = state_dict["pretrained.patch_embed.proj.weight"]
    # TTNN conv expects [out_channels, in_channels, kH, kW]
    parameters["patch_embed_weight"] = ttnn.from_torch(
        patch_embed_weight.unsqueeze(0).transpose(-2, -1),  # [1, 1024, 3, 14]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # CLS token: [1, 1, 1024]
    cls_token = state_dict["pretrained.cls_token"]
    parameters["cls_token"] = ttnn.from_torch(
        cls_token, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Positional embedding: [1, num_patches+1, 1024]
    pos_embed = state_dict["pretrained.pos_embed"]
    parameters["pos_embed"] = ttnn.from_torch(
        pos_embed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    return parameters


def preprocess_encoder_weights(
    model, device: ttnn.Device, config: DepthAnythingV2Config
) -> List[Dict[str, ttnn.Tensor]]:
    """Preprocess all 24 ViT encoder block weights for TTNN."""
    state_dict = model.state_dict()
    blocks = []

    for i in range(config.num_layers):
        block_params = {}
        prefix = f"pretrained.blocks.{i}."

        # LayerNorm 1 (attention pre-norm)
        ln1_weight = state_dict[prefix + "norm1.weight"]
        ln1_bias = state_dict[prefix + "norm1.bias"]
        block_params["norm1_weight"] = ttnn.from_torch(
            ln1_weight.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        block_params["norm1_bias"] = ttnn.from_torch(
            ln1_bias.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Attention QKV: single weight [3*1024, 1024] + bias [3*1024]
        qkv_weight = state_dict[prefix + "attn.qkv.weight"]
        qkv_bias = state_dict[prefix + "attn.qkv.weight"]  # same key, we handle below
        # Actually, DINOv2 stores QKV as a combined linear layer
        # qkv.weight shape: [3072, 1024] (3 * 1024)
        # qkv.bias shape: [3072]
        block_params["qkv_weight"] = ttnn.from_torch(
            qkv_weight.transpose(0, 1).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if prefix + "attn.qkv.bias" in state_dict:
            qkv_bias = state_dict[prefix + "attn.qkv.bias"]
            block_params["qkv_bias"] = ttnn.from_torch(
                qkv_bias.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Attention proj: [1024, 1024]
        proj_weight = state_dict[prefix + "attn.proj.weight"]
        block_params["proj_weight"] = ttnn.from_torch(
            proj_weight.transpose(0, 1).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if prefix + "attn.proj.bias" in state_dict:
            proj_bias = state_dict[prefix + "attn.proj.bias"]
            block_params["proj_bias"] = ttnn.from_torch(
                proj_bias.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # LayerScale for attention (gamma1): [1024]
        if prefix + "ls1.gamma" in state_dict:
            ls1_gamma = state_dict[prefix + "ls1.gamma"]
            block_params["ls1_gamma"] = ttnn.from_torch(
                ls1_gamma.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # LayerNorm 2 (MLP pre-norm)
        ln2_weight = state_dict[prefix + "norm2.weight"]
        ln2_bias = state_dict[prefix + "norm2.bias"]
        block_params["norm2_weight"] = ttnn.from_torch(
            ln2_weight.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        block_params["norm2_bias"] = ttnn.from_torch(
            ln2_bias.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # MLP fc1: [4096, 1024]
        mlp_fc1_weight = state_dict[prefix + "mlp.fc1.weight"]
        block_params["mlp_fc1_weight"] = ttnn.from_torch(
            mlp_fc1_weight.transpose(0, 1).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if prefix + "mlp.fc1.bias" in state_dict:
            mlp_fc1_bias = state_dict[prefix + "mlp.fc1.bias"]
            block_params["mlp_fc1_bias"] = ttnn.from_torch(
                mlp_fc1_bias.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # MLP fc2: [1024, 4096]
        mlp_fc2_weight = state_dict[prefix + "mlp.fc2.weight"]
        block_params["mlp_fc2_weight"] = ttnn.from_torch(
            mlp_fc2_weight.transpose(0, 1).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if prefix + "mlp.fc2.bias" in state_dict:
            mlp_fc2_bias = state_dict[prefix + "mlp.fc2.bias"]
            block_params["mlp_fc2_bias"] = ttnn.from_torch(
                mlp_fc2_bias.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # LayerScale for MLP (gamma2): [1024]
        if prefix + "ls2.gamma" in state_dict:
            ls2_gamma = state_dict[prefix + "ls2.gamma"]
            block_params["ls2_gamma"] = ttnn.from_torch(
                ls2_gamma.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        blocks.append(block_params)

    return blocks


def preprocess_final_norm(model, device: ttnn.Device) -> Dict[str, ttnn.Tensor]:
    """Preprocess final LayerNorm weights."""
    state_dict = model.state_dict()
    params = {}
    params["norm_weight"] = ttnn.from_torch(
        state_dict["pretrained.norm.weight"].unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    params["norm_bias"] = ttnn.from_torch(
        state_dict["pretrained.norm.bias"].unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return params


def preprocess_dpt_head_weights(
    model, device: ttnn.Device, config: DepthAnythingV2Config
) -> Dict[str, ttnn.Tensor]:
    """Preprocess DPT decoder head weights for TTNN.

    The DPT head consists of:
    - 4 projection convolutions (1x1 Conv2d: 1024 -> out_channels[i])
    - 4 resize layers (ConvTranspose2d or Conv2d or Identity)
    - 4 layer_rn convolutions (3x3 Conv2d)
    - 4 fusion blocks (each with 2 ResidualConvUnits + 1x1 conv)
    - 2 output convolutions
    """
    state_dict = model.state_dict()
    params = {}
    prefix = "depth_head."

    # Projection layers: 1x1 Conv2d for each intermediate feature
    for i, out_ch in enumerate(config.dpt_out_channels):
        proj_w = state_dict[f"{prefix}projects.{i}.weight"]  # [out_ch, 1024, 1, 1]
        proj_b = state_dict[f"{prefix}projects.{i}.bias"]  # [out_ch]
        params[f"proj_{i}_weight"] = ttnn.from_torch(
            proj_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        params[f"proj_{i}_bias"] = ttnn.from_torch(
            proj_b.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Resize layers (kept as PyTorch for now, processed in forward pass)
    for i in range(4):
        layer_key = f"{prefix}resize_layers.{i}"
        if f"{layer_key}.weight" in state_dict:
            params[f"resize_{i}_weight"] = ttnn.from_torch(
                state_dict[f"{layer_key}.weight"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if f"{layer_key}.bias" in state_dict:
                params[f"resize_{i}_bias"] = ttnn.from_torch(
                    state_dict[f"{layer_key}.bias"],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

    # Layer RN convolutions: 3x3 Conv2d
    for i in range(4):
        rn_w = state_dict[f"{prefix}scratch.layer{i+1}_rn.weight"]
        params[f"layer_rn_{i}_weight"] = ttnn.from_torch(
            rn_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if f"{prefix}scratch.layer{i+1}_rn.bias" in state_dict:
            rn_b = state_dict[f"{prefix}scratch.layer{i+1}_rn.bias"]
            params[f"layer_rn_{i}_bias"] = ttnn.from_torch(
                rn_b.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    # RefineNet fusion blocks (4 blocks, each with 2 ResidualConvUnits + out_conv)
    for i in range(1, 5):
        block_prefix = f"{prefix}scratch.refinenet{i}."
        # ResidualConvUnit 1: conv1, conv2
        for j in range(1, 3):
            rcu_conv_w = state_dict.get(f"{block_prefix}resConfUnit{j}.conv{j}.weight")
            if rcu_conv_w is not None:
                params[f"refinenet{i}_rcu{j}_conv{j}_weight"] = ttnn.from_torch(
                    rcu_conv_w,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                rcu_conv_b = state_dict.get(f"{block_prefix}resConfUnit{j}.conv{j}.bias")
                if rcu_conv_b is not None:
                    params[f"refinenet{i}_rcu{j}_conv{j}_bias"] = ttnn.from_torch(
                        rcu_conv_b.unsqueeze(0).unsqueeze(0),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
        # out_conv: 1x1 Conv2d
        out_conv_w = state_dict.get(f"{block_prefix}out_conv.weight")
        if out_conv_w is not None:
            params[f"refinenet{i}_out_conv_weight"] = ttnn.from_torch(
                out_conv_w,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out_conv_b = state_dict.get(f"{block_prefix}out_conv.bias")
            if out_conv_b is not None:
                params[f"refinenet{i}_out_conv_bias"] = ttnn.from_torch(
                    out_conv_b.unsqueeze(0).unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

    # Output convolutions
    out_conv1_w = state_dict[f"{prefix}scratch.output_conv1.weight"]
    params["output_conv1_weight"] = ttnn.from_torch(
        out_conv1_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_conv1_b = state_dict.get(f"{prefix}scratch.output_conv1.bias")
    if out_conv1_b is not None:
        params["output_conv1_bias"] = ttnn.from_torch(
            out_conv1_b.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # output_conv2 is a Sequential with Conv2d + ReLU + Conv2d + ReLU + Identity
    for k, name in enumerate(["0", "2"]):
        w = state_dict.get(f"{prefix}scratch.output_conv2.{name}.weight")
        if w is not None:
            params[f"output_conv2_{k}_weight"] = ttnn.from_torch(
                w,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            b = state_dict.get(f"{prefix}scratch.output_conv2.{name}.bias")
            if b is not None:
                params[f"output_conv2_{k}_bias"] = ttnn.from_torch(
                    b.unsqueeze(0).unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

    return params


def preprocess_all_weights_for_ttnn(
    model, device: ttnn.Device, config: DepthAnythingV2Config
) -> Dict:
    """Preprocess all model weights for TTNN inference."""
    parameters = {}
    parameters["patch_embed"] = preprocess_patch_embed_weights(model, device, config)
    parameters["encoder_blocks"] = preprocess_encoder_weights(model, device, config)
    parameters["final_norm"] = preprocess_final_norm(model, device)
    parameters["dpt_head"] = preprocess_dpt_head_weights(model, device, config)
    return parameters


# =============================================================================
# TTNN Layer Implementations
# =============================================================================


def ttnn_layer_norm(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    epsilon: float = 1e-6,
) -> ttnn.Tensor:
    """Apply LayerNorm using TTNN."""
    return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=epsilon)


def ttnn_attention(
    x: ttnn.Tensor,
    qkv_weight: ttnn.Tensor,
    qkv_bias: Optional[ttnn.Tensor],
    proj_weight: ttnn.Tensor,
    proj_bias: Optional[ttnn.Tensor],
    num_heads: int,
    head_dim: int,
    config: DepthAnythingV2Config,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """Multi-head self-attention using TTNN.

    Args:
        x: Input tensor [1, 1, seq_len, embed_dim]
        qkv_weight: QKV projection weight [1, 1, embed_dim, 3*embed_dim]
        qkv_bias: QKV projection bias [1, 1, 1, 3*embed_dim]
        proj_weight: Output projection weight [1, 1, embed_dim, embed_dim]
        proj_bias: Output projection bias [1, 1, 1, embed_dim]
        num_heads: Number of attention heads
        head_dim: Dimension per head
        config: Model configuration
        compute_kernel_config: Compute kernel config
    """
    seq_len = x.shape[-2]
    embed_dim = num_heads * head_dim
    memory_config = config.get_memory_config()

    # QKV projection: x @ W_qkv + b_qkv -> [1, 1, seq_len, 3*embed_dim]
    qkv = ttnn.linear(
        x,
        qkv_weight,
        bias=qkv_bias,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )

    # Split QKV: [seq_len, 3*embed_dim] -> 3 x [seq_len, embed_dim]
    # Then reshape to [num_heads, seq_len, head_dim]
    qkv_torch = ttnn.to_torch(qkv)
    q, k, v = qkv_torch.chunk(3, dim=-1)

    # Reshape for multi-head: [1, seq_len, num_heads, head_dim] -> [1, num_heads, seq_len, head_dim]
    q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

    # Scale dot-product attention
    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    # Reshape back: [1, num_heads, seq_len, head_dim] -> [1, seq_len, embed_dim]
    attn_output = attn_output.transpose(1, 2).reshape(1, seq_len, embed_dim)

    # Move back to device for output projection
    attn_output_ttnn = ttnn.from_torch(
        attn_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x.device(), memory_config=memory_config
    )

    # Output projection
    output = ttnn.linear(
        attn_output_ttnn,
        proj_weight,
        bias=proj_bias,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )

    return output


def ttnn_mlp(
    x: ttnn.Tensor,
    fc1_weight: ttnn.Tensor,
    fc1_bias: Optional[ttnn.Tensor],
    fc2_weight: ttnn.Tensor,
    fc2_bias: Optional[ttnn.Tensor],
    config: DepthAnythingV2Config,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """MLP block: fc1 -> GELU -> fc2 using TTNN."""
    memory_config = config.get_memory_config()

    # fc1: [embed_dim] -> [4*embed_dim]
    hidden = ttnn.linear(
        x,
        fc1_weight,
        bias=fc1_bias,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )

    # GELU activation
    hidden = ttnn.gelu(hidden, memory_config=memory_config)

    # fc2: [4*embed_dim] -> [embed_dim]
    output = ttnn.linear(
        hidden,
        fc2_weight,
        bias=fc2_bias,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )

    return output


def ttnn_encoder_block(
    x: ttnn.Tensor,
    block_params: Dict[str, ttnn.Tensor],
    layer_idx: int,
    config: DepthAnythingV2Config,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """Single ViT encoder block with LayerScale.

    Architecture:
        x = x + ls1(attention(norm1(x)))
        x = x + ls2(mlp(norm2(x)))
    """
    memory_config = config.get_memory_config()

    # --- Attention Branch ---
    # Pre-norm
    normed = ttnn_layer_norm(
        x,
        block_params["norm1_weight"],
        block_params["norm1_bias"],
        epsilon=config.layer_norm_eps,
    )

    # Self-attention
    attn_out = ttnn_attention(
        normed,
        block_params["qkv_weight"],
        block_params.get("qkv_bias"),
        block_params["proj_weight"],
        block_params.get("proj_bias"),
        config.num_heads,
        config.head_dim,
        config,
        compute_kernel_config,
    )

    # LayerScale: multiply by learnable scale parameter
    if "ls1_gamma" in block_params:
        attn_out = ttnn.multiply(attn_out, block_params["ls1_gamma"])

    # Residual connection
    x = ttnn.add(x, attn_out)

    # --- MLP Branch ---
    # Pre-norm
    normed = ttnn_layer_norm(
        x,
        block_params["norm2_weight"],
        block_params["norm2_bias"],
        epsilon=config.layer_norm_eps,
    )

    # MLP
    mlp_out = ttnn_mlp(
        normed,
        block_params["mlp_fc1_weight"],
        block_params.get("mlp_fc1_bias"),
        block_params["mlp_fc2_weight"],
        block_params.get("mlp_fc2_bias"),
        config,
        compute_kernel_config,
    )

    # LayerScale
    if "ls2_gamma" in block_params:
        mlp_out = ttnn.multiply(mlp_out, block_params["ls2_gamma"])

    # Residual connection
    x = ttnn.add(x, mlp_out)

    return x


# =============================================================================
# Full Model Forward Pass
# =============================================================================


def run_patch_embedding(
    pixel_values: torch.Tensor,
    parameters: Dict[str, ttnn.Tensor],
    device: ttnn.Device,
    config: DepthAnythingV2Config,
) -> ttnn.Tensor:
    """Run patch embedding: Conv2d(3, 1024, 14, 14) + CLS token + pos embed.

    Args:
        pixel_values: Input image [B, 3, 518, 518]
        parameters: Preprocessed weights dict
        device: TTNN device
        config: Model config

    Returns:
        Embedded tokens [1, 1, num_patches+1, 1024]
    """
    B, C, H, W = pixel_values.shape
    patch_h = H // config.patch_size  # 37
    patch_w = W // config.patch_size  # 37
    num_patches = patch_h * patch_w  # 1369

    # Patch embedding via unfold + linear (equivalent to Conv2d with stride=patch_size)
    # Unfold image to patches: [B, 3*14*14, num_patches]
    patches = F.unfold(pixel_values, kernel_size=config.patch_size, stride=config.patch_size)
    # Reshape to [B, num_patches, 3*14*14]
    patches = patches.transpose(1, 2)

    # Project patches: [B, num_patches, 3*14*14] @ [3*14*14, 1024] -> [B, num_patches, 1024]
    # We'll do this with the conv weight reshaped
    conv_weight = ttnn.to_torch(parameters["patch_embed_weight"])  # already on device
    # The original conv weight is [1024, 3, 14, 14], reshape to [1024, 3*14*14] for linear
    conv_w_linear = conv_weight.reshape(1, 1, config.embed_dim, -1)

    # Prepend CLS token
    cls_token = ttnn.to_torch(parameters["cls_token"])  # [1, 1, 1024]
    cls_tokens = cls_token.expand(B, -1, -1)  # [B, 1, 1024]
    embeddings = torch.cat([cls_tokens, patches], dim=1)  # [B, num_patches+1, 1024]

    # Add positional embedding
    pos_embed = ttnn.to_torch(parameters["pos_embed"])  # [1, num_patches+1, 1024]
    embeddings = embeddings + pos_embed

    # Move to device
    embeddings_ttnn = ttnn.from_torch(
        embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    return embeddings_ttnn


def run_vit_encoder(
    x: ttnn.Tensor,
    encoder_blocks: List[Dict[str, ttnn.Tensor]],
    config: DepthAnythingV2Config,
    compute_kernel_config=None,
) -> Tuple[ttnn.Tensor, List[ttnn.Tensor]]:
    """Run ViT-L/14 encoder and extract intermediate features.

    Returns:
        - Final normalized features
        - List of intermediate features at specified layer indices
    """
    intermediate_outputs = []
    intermediate_set = set(config.intermediate_layer_indices)

    for i in range(config.num_layers):
        x = ttnn_encoder_block(x, encoder_blocks[i], i, config, compute_kernel_config)

        if i in intermediate_set:
            intermediate_outputs.append(x)

    return x, intermediate_outputs


def run_dpt_head_on_cpu(
    intermediate_features: List[ttnn.Tensor],
    final_norm_params: Dict[str, ttnn.Tensor],
    dpt_params: Dict[str, ttnn.Tensor],
    model,
    device: ttnn.Device,
    config: DepthAnythingV2Config,
) -> torch.Tensor:
    """Run DPT decoder head.

    For Stage 1 bring-up, the DPT head runs on CPU using PyTorch reference.
    This can be migrated to TTNN in Stage 2 optimization.

    Args:
        intermediate_features: List of 4 intermediate feature tensors from ViT encoder
        final_norm_params: Final LayerNorm parameters
        dpt_params: DPT head parameters
        model: Original PyTorch model (for reference DPT head)
        device: TTNN device
        config: Model config

    Returns:
        Depth map tensor [1, H, W]
    """
    patch_h = config.patch_h
    patch_w = config.patch_w

    # Move intermediate features to CPU and apply final norm
    cpu_features = []
    for feat_ttnn in intermediate_features:
        feat = ttnn.to_torch(feat_ttnn)
        # Apply final LayerNorm using PyTorch for accuracy
        norm_w = ttnn.to_torch(final_norm_params["norm_weight"]).squeeze()
        norm_b = ttnn.to_torch(final_norm_params["norm_bias"]).squeeze()
        feat = F.layer_norm(feat.float(), [config.embed_dim], norm_w.float(), norm_b.float())
        # Extract patch tokens (skip CLS token at position 0)
        feat = feat[:, 1:, :]  # [B, num_patches, embed_dim]
        cpu_features.append((feat, None))  # (patch_features, cls_token) - no cls token for Large

    # Run DPT head using PyTorch reference model
    with torch.no_grad():
        depth = model.depth_head(cpu_features, patch_h, patch_w)
        depth = F.relu(depth)

    return depth.squeeze(1)


def run_depth_anything_v2_inference(
    pixel_values: torch.Tensor,
    parameters: Dict,
    model,
    device: ttnn.Device,
    config: DepthAnythingV2Config,
) -> torch.Tensor:
    """Full Depth-Anything-V2-Large inference pipeline.

    Args:
        pixel_values: Input image tensor [B, 3, 518, 518] (normalized)
        parameters: All preprocessed TTNN weights
        model: Original PyTorch model (for DPT head reference)
        device: TTNN device
        config: Model configuration

    Returns:
        Depth map tensor [B, H, W]
    """
    compute_kernel_config = config.get_compute_kernel_config(device)

    # Step 1: Patch Embedding
    x = run_patch_embedding(pixel_values, parameters["patch_embed"], device, config)

    # Step 2: ViT Encoder (24 blocks) with intermediate feature extraction
    x, intermediate_features = run_vit_encoder(
        x, parameters["encoder_blocks"], config, compute_kernel_config
    )

    # Step 3: DPT Decoder Head (CPU for Stage 1)
    depth = run_dpt_head_on_cpu(
        intermediate_features,
        parameters["final_norm"],
        parameters["dpt_head"],
        model,
        device,
        config,
    )

    return depth
