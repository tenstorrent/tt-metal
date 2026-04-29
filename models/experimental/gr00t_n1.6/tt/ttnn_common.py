# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Common TTNN utilities for GR00T N1.6 implementation.
"""

import torch
import ttnn


def nearest_32(x: int) -> int:
    """Round up to nearest multiple of 32 for TTNN tile alignment."""
    return ((x + 31) // 32) * 32


def to_tt_tensor(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Convert PyTorch tensor to TTNN tensor on device."""
    return ttnn.from_torch(
        tensor.to(torch.bfloat16) if tensor.dtype != torch.bfloat16 else tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def preprocess_linear_weight(
    weight: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
) -> ttnn.Tensor:
    """Preprocess linear weight for TTNN: transpose and convert."""
    # TTNN linear expects weight in [in_features, out_features] format
    w = weight.t().contiguous().to(torch.bfloat16)
    return ttnn.from_torch(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def preprocess_linear_bias(
    bias: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """Preprocess linear bias for TTNN."""
    b = bias.unsqueeze(0).to(torch.bfloat16)
    return ttnn.from_torch(
        b,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def preprocess_layernorm_params(
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: ttnn.Device,
) -> tuple:
    """Preprocess LayerNorm weight and bias for TTNN."""
    w = ttnn.from_torch(
        weight.unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    b = ttnn.from_torch(
        bias.unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return w, b


def preprocess_rmsnorm_params(
    weight: torch.Tensor,
    device: ttnn.Device,
) -> ttnn.Tensor:
    """Preprocess RMSNorm weight for TTNN."""
    return ttnn.from_torch(
        weight.unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


# Default core grid for Blackhole p150a
CORE_GRID_BH = ttnn.CoreGrid(y=8, x=8)
