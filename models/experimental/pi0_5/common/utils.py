# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utility functions for TTNN PI0 implementation.

This module provides shared helper functions used across the PI0 model:
    - Sinusoidal positional embeddings for flow matching timesteps
    - Safe tensor operations with dtype handling
    - Device-aware computations
"""

import math
from typing import Optional

import torch

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


def get_ttnn_dtype(precision: str) -> "ttnn.DataType":
    """
    Convert precision string to TTNN dtype.

    Args:
        precision: "bfloat16", "float32", "bfloat8_b", etc.

    Returns:
        TTNN data type
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")

    dtype_map = {
        "bfloat16": ttnn.bfloat16,
        "float32": ttnn.float32,
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": ttnn.bfloat4_b,
    }
    return dtype_map.get(precision, ttnn.bfloat16)


def create_sinusoidal_pos_embedding_torch(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings for timesteps (PyTorch fallback).

    This is used for flow matching where we need to encode continuous
    timestep values [0, 1] into high-dimensional embeddings.

    Args:
        time: Tensor of shape (batch_size,) with timestep values
        dimension: Embedding dimension (must be divisible by 2)
        min_period: Minimum period for sinusoidal encoding
        max_period: Maximum period for sinusoidal encoding

    Returns:
        Tensor of shape (batch_size, dimension) with sinusoidal embeddings
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("time tensor must be 1D (batch_size,)")

    device = time.device
    dtype = torch.float64 if device.type == "cpu" else time.dtype

    # Create frequency fractions [0, 1] for dimension//2 frequencies
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)

    # Compute periods using exponential spacing
    period = min_period * (max_period / min_period) ** fraction

    # Compute scaling factor: 2π / period
    scaling_factor = (1.0 / period) * 2 * math.pi

    # Outer product: scaling_factor[None, :] * time[:, None]
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)

    # Concatenate sin and cos embeddings
    embeddings = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

    return embeddings.to(time.dtype)


def create_sinusoidal_pos_embedding_ttnn(
    time: "ttnn.Tensor",
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: Optional["ttnn.Device"] = None,
) -> "ttnn.Tensor":
    """
    Create sinusoidal positional embeddings for timesteps (TTNN version).

    Note: The frequency computation is done on host (torch) since it's
    a one-time computation. The sin/cos operations are done on device.

    Args:
        time: TTNN tensor of shape (batch_size,) with timestep values
        dimension: Embedding dimension (must be divisible by 2)
        min_period: Minimum period for sinusoidal encoding
        max_period: Maximum period for sinusoidal encoding
        device: TTNN device (uses time's device if not specified)

    Returns:
        TTNN tensor of shape (batch_size, dimension) with sinusoidal embeddings
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")

    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if device is None:
        device = time.device()

    # Compute frequencies on host (one-time computation)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = (1.0 / period) * 2 * math.pi

    # Convert scaling factor to TTNN
    scaling_factor_ttnn = ttnn.from_torch(
        scaling_factor.unsqueeze(0),  # [1, dim//2]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Reshape time for broadcasting: [batch, 1]
    time_reshaped = ttnn.reshape(time, (-1, 1))

    # Compute sin input: time * scaling_factor
    sin_input = ttnn.matmul(time_reshaped, scaling_factor_ttnn)

    # Compute sin and cos
    sin_emb = ttnn.sin(sin_input)
    cos_emb = ttnn.cos(sin_input)

    # Concatenate
    embeddings = ttnn.concat([sin_emb, cos_emb], dim=-1)

    return embeddings


def safe_cat_torch(
    tensors: list,
    dim: int = -1,
) -> torch.Tensor:
    """
    Safely concatenate tensors with dtype handling.

    Converts all tensors to the first tensor's dtype before concatenation.

    Args:
        tensors: List of tensors to concatenate
        dim: Dimension along which to concatenate

    Returns:
        Concatenated tensor
    """
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")

    target_dtype = tensors[0].dtype
    converted = [t.to(dtype=target_dtype) if t.dtype != target_dtype else t for t in tensors]
    return torch.cat(converted, dim=dim)


def safe_cat_ttnn(
    tensors: list,
    dim: int = -1,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
) -> "ttnn.Tensor":
    """
    Safely concatenate TTNN tensors.

    Args:
        tensors: List of TTNN tensors to concatenate
        dim: Dimension along which to concatenate
        memory_config: Optional memory config for output

    Returns:
        Concatenated TTNN tensor
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")

    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")

    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    return ttnn.concat(tensors, dim=dim, memory_config=memory_config)


def compute_position_ids_torch(pad_masks: torch.Tensor) -> torch.Tensor:
    """
    Compute position IDs from padding masks.

    Args:
        pad_masks: Boolean tensor (batch_size, seq_len) where True = valid token

    Returns:
        Position IDs tensor (batch_size, seq_len)
    """
    return torch.cumsum(pad_masks.long(), dim=1) - 1


def compute_position_ids_ttnn(
    pad_masks: "ttnn.Tensor",
    device: Optional["ttnn.Device"] = None,
) -> "ttnn.Tensor":
    """
    Compute position IDs from padding masks (TTNN version).

    Args:
        pad_masks: Boolean TTNN tensor (batch_size, seq_len)
        device: TTNN device

    Returns:
        Position IDs TTNN tensor (batch_size, seq_len)
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")

    # Use moreh_cumsum for cumulative sum
    cumsum = ttnn.moreh_cumsum(pad_masks, dim=1)

    # Subtract 1 to get 0-indexed positions
    ones = ttnn.ones_like(cumsum)
    position_ids = ttnn.subtract(cumsum, ones)

    return position_ids


def ttnn_to_torch(tensor: "ttnn.Tensor") -> torch.Tensor:
    """
    Convert TTNN tensor to PyTorch tensor.

    Args:
        tensor: TTNN tensor

    Returns:
        PyTorch tensor
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")

    return ttnn.to_torch(tensor)


def torch_to_ttnn(
    tensor: torch.Tensor,
    device: "ttnn.Device",
    dtype: Optional["ttnn.DataType"] = None,
    layout: Optional["ttnn.Layout"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
) -> "ttnn.Tensor":
    """
    Convert PyTorch tensor to TTNN tensor.

    Args:
        tensor: PyTorch tensor
        device: TTNN device
        dtype: TTNN data type (default: bfloat16)
        layout: TTNN layout (default: TILE_LAYOUT)
        memory_config: Memory configuration (default: DRAM)

    Returns:
        TTNN tensor
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")

    if dtype is None:
        dtype = ttnn.bfloat16
    if layout is None:
        layout = ttnn.TILE_LAYOUT
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


# Export both torch and ttnn versions
# Use torch versions as fallback when TTNN is not available
create_sinusoidal_pos_embedding = create_sinusoidal_pos_embedding_torch
safe_cat = safe_cat_torch
compute_position_ids = compute_position_ids_torch
sample_noise = sample_noise_torch
sample_time = sample_time_torch
