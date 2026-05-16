# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Centralized TTNN tensor conversion and memory utilities for RVC.

Provides a single source of truth for:
    - torch → TTNN tensor conversion
    - weight preprocessing patterns
    - memory config defaults
    - conv output postprocessing

All tensor format conventions:
    Linear weights: [in_features, out_features] (pre-transposed from PyTorch)
    Linear bias: [1, 1, out_features]
    LayerNorm weight/bias: [1, 1, channels]
    Conv1d weights: [out_ch, in_ch/groups, kernel_size] (PyTorch format, ROW_MAJOR)
    Activations: [B, S, C] channels-last for linear/attention
    Conv inputs: [B, L, C] channels-last
"""

import torch
import ttnn
from typing import Optional


# =====================================================================
# Default memory configs (Stage 1: all DRAM, no sharding)
# =====================================================================

DEFAULT_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
DEFAULT_DTYPE = ttnn.bfloat16
DEFAULT_LAYOUT = ttnn.TILE_LAYOUT


# =====================================================================
# Tensor conversion helpers
# =====================================================================

def to_device(
    tensor: torch.Tensor,
    device,
    dtype=DEFAULT_DTYPE,
    layout=DEFAULT_LAYOUT,
    memory_config=DEFAULT_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """
    Convert a torch tensor to TTNN on device.

    This is the single canonical way to place tensors on device in RVC.

    Args:
        tensor: PyTorch tensor (any dtype, will be cast to float32 first).
        device: TTNN device.
        dtype: TTNN dtype (default: bfloat16).
        layout: TTNN layout (default: TILE_LAYOUT).
        memory_config: TTNN memory config (default: DRAM).

    Returns:
        TTNN tensor on device.
    """
    return ttnn.from_torch(
        tensor.float(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def to_host(
    tensor: ttnn.Tensor,
) -> torch.Tensor:
    """
    Convert a TTNN tensor back to torch on host.

    Args:
        tensor: TTNN tensor on device.

    Returns:
        PyTorch float32 tensor.
    """
    return ttnn.to_torch(tensor).float()


# =====================================================================
# Weight preprocessing helpers
# =====================================================================

def preprocess_linear_weight(
    weight: torch.Tensor,
    device,
) -> ttnn.Tensor:
    """
    Preprocess PyTorch Linear weight for TTNN.

    PyTorch: [out_features, in_features]
    TTNN:    [in_features, out_features] (transposed)

    Args:
        weight: PyTorch Linear weight tensor.
        device: TTNN device.

    Returns:
        TTNN tensor in TILE_LAYOUT on device.
    """
    return to_device(weight.T.contiguous(), device)


def preprocess_linear_bias(
    bias: torch.Tensor,
    device,
) -> ttnn.Tensor:
    """
    Preprocess PyTorch Linear bias for TTNN.

    PyTorch: [out_features]
    TTNN:    [1, 1, out_features]

    Args:
        bias: PyTorch Linear bias tensor.
        device: TTNN device.

    Returns:
        TTNN tensor in TILE_LAYOUT on device.
    """
    return to_device(bias.unsqueeze(0).unsqueeze(0), device)


def preprocess_linear(
    linear: torch.nn.Linear,
    device,
) -> dict:
    """
    Preprocess a full PyTorch Linear module for TTNN.

    Returns:
        Dict with 'weight' and 'bias' keys.
    """
    result = {"weight": preprocess_linear_weight(linear.weight, device)}
    if linear.bias is not None:
        result["bias"] = preprocess_linear_bias(linear.bias, device)
    else:
        result["bias"] = None
    return result


def preprocess_layer_norm(
    layer_norm: torch.nn.LayerNorm,
    device,
) -> dict:
    """
    Preprocess a PyTorch LayerNorm for TTNN.

    Returns:
        Dict with 'weight' and 'bias' keys.
    """
    return {
        "weight": to_device(layer_norm.weight.unsqueeze(0).unsqueeze(0), device),
        "bias": to_device(layer_norm.bias.unsqueeze(0).unsqueeze(0), device),
    }


def preprocess_conv1d_weight(
    weight: torch.Tensor,
) -> ttnn.Tensor:
    """
    Preprocess Conv1d weight for TTNN (host-side, ROW_MAJOR).

    PyTorch Conv1d weight: [out_ch, in_ch/groups, kernel_size]
    TTNN expects same format but in ROW_MAJOR layout on host.

    Args:
        weight: PyTorch Conv1d weight tensor.

    Returns:
        TTNN tensor in ROW_MAJOR_LAYOUT on host.
    """
    return ttnn.from_torch(
        weight.float(), dtype=DEFAULT_DTYPE, layout=ttnn.ROW_MAJOR_LAYOUT,
    )


# =====================================================================
# Conv output postprocessing
# =====================================================================

def postprocess_conv_output(
    output_tensor: ttnn.Tensor,
    batch_size: int,
    output_length: int,
    out_channels: int,
    to_nchw: bool = True,
) -> torch.Tensor:
    """
    Convert TTNN conv output to torch tensor.

    Conv1d/ConvTranspose1d outputs are in [1, 1, L_out, C_out] NHWC format.
    This function extracts and optionally converts to [B, C, L] format.

    Args:
        output_tensor: TTNN tensor from conv op (may be sharded).
        batch_size: Batch size.
        output_length: Output sequence length.
        out_channels: Number of output channels.
        to_nchw: If True, return [B, C, L]. If False, return [B, L, C].

    Returns:
        torch.Tensor in requested format.
    """
    # Handle sharded output
    try:
        output_tensor = ttnn.sharded_to_interleaved(output_tensor)
    except RuntimeError:
        pass

    out = ttnn.to_torch(ttnn.from_device(output_tensor)).float()
    out = out.reshape(batch_size, 1, output_length, -1)[:, :, :, :out_channels].squeeze(1)

    if to_nchw:
        out = out.permute(0, 2, 1)  # [B, C, L]

    return out
