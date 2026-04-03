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
import ttnn


def get_ttnn_dtype(precision: str) -> ttnn.DataType:
    """
    Convert precision string to TTNN dtype.

    Args:
        precision: "bfloat16", "float32", "bfloat8_b", etc.

    Returns:
        TTNN data type
    """
    dtype_map = {
        "bfloat16": ttnn.bfloat16,
        "float32": ttnn.float32,
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": ttnn.bfloat4_b,
    }
    return dtype_map.get(precision, ttnn.bfloat16)


def precompute_sinusoidal_scaling_factor(
    dimension: int,
    min_period: float,
    max_period: float,
    device: ttnn.Device,
    indices: ttnn.Tensor,
) -> ttnn.Tensor:
    """Pre-compute the scaling factor for sinusoidal embeddings (constant across timesteps)."""
    half_dim = dimension // 2
    indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)
    if half_dim > 1:
        fraction = ttnn.multiply(indices, 1.0 / (half_dim - 1))
    else:
        fraction = indices

    log_ratio = math.log(max_period / min_period)
    exponent = ttnn.multiply(fraction, log_ratio)
    period_ratio = ttnn.exp(exponent)
    period = ttnn.multiply(period_ratio, min_period)
    inv_period = ttnn.reciprocal(period)
    scaling_factor = ttnn.multiply(inv_period, 2 * math.pi)
    scaling_factor = ttnn.reshape(scaling_factor, (1, half_dim))

    # Cleanup intermediates
    ttnn.deallocate(fraction)
    ttnn.deallocate(exponent)
    ttnn.deallocate(period_ratio)
    ttnn.deallocate(period)
    ttnn.deallocate(inv_period)

    return scaling_factor


def create_sinusoidal_pos_embedding_ttnn(
    time: ttnn.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: Optional[ttnn.Device] = None,
    indices: Optional[ttnn.Tensor] = None,
    precomputed_scaling_factor: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    """
    Create sinusoidal positional embeddings for timesteps (pure TTNN version).

    If precomputed_scaling_factor is provided, skips the 8-op scaling factor
    computation (saves ~80 ops per 10-step inference).
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if device is None:
        device = time.device()

    half_dim = dimension // 2

    if precomputed_scaling_factor is not None:
        scaling_factor = precomputed_scaling_factor
    else:
        # Compute scaling factor on-the-fly (slow path)
        indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)
        if half_dim > 1:
            fraction = ttnn.multiply(indices, 1.0 / (half_dim - 1))
        else:
            fraction = indices
        log_ratio = math.log(max_period / min_period)
        exponent = ttnn.multiply(fraction, log_ratio)
        period_ratio = ttnn.exp(exponent)
        period = ttnn.multiply(period_ratio, min_period)
        inv_period = ttnn.reciprocal(period)
        scaling_factor = ttnn.multiply(inv_period, 2 * math.pi)
        scaling_factor = ttnn.reshape(scaling_factor, (1, half_dim))
        ttnn.deallocate(fraction)
        ttnn.deallocate(exponent)
        ttnn.deallocate(period_ratio)
        ttnn.deallocate(period)
        ttnn.deallocate(inv_period)

    # Only 4 ops per call when scaling_factor is pre-computed:
    # reshape + matmul + sin + cos + concat
    time_reshaped = ttnn.reshape(time, (-1, 1))
    sin_input = ttnn.matmul(time_reshaped, scaling_factor)
    sin_emb = ttnn.sin(sin_input)
    cos_emb = ttnn.cos(sin_input)
    embeddings = ttnn.concat([sin_emb, cos_emb], dim=-1)

    ttnn.deallocate(sin_input)

    return embeddings


def safe_cat_ttnn(
    tensors: list,
    dim: int = -1,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    Safely concatenate TTNN tensors.

    Args:
        tensors: List of TTNN tensors to concatenate
        dim: Dimension along which to concatenate
        memory_config: Optional memory config for output

    Returns:
        Concatenated TTNN tensor
    """
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")

    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    return ttnn.concat(tensors, dim=dim, memory_config=memory_config)


def compute_position_ids_ttnn(
    pad_masks: ttnn.Tensor,
    device: Optional[ttnn.Device] = None,
) -> ttnn.Tensor:
    """
    Compute position IDs from padding masks (TTNN version).

    Args:
        pad_masks: Boolean TTNN tensor (batch_size, seq_len)
        device: TTNN device

    Returns:
        Position IDs TTNN tensor (batch_size, seq_len)
    """
    # Use moreh_cumsum for cumulative sum
    cumsum = ttnn.moreh_cumsum(pad_masks, dim=1)

    # Subtract 1 to get 0-indexed positions
    ones = ttnn.ones_like(cumsum)
    position_ids = ttnn.subtract(cumsum, ones)

    return position_ids


def ttnn_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    """
    Convert TTNN tensor to PyTorch tensor.

    Args:
        tensor: TTNN tensor

    Returns:
        PyTorch tensor
    """
    return ttnn.to_torch(tensor)


def torch_to_ttnn(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: Optional[ttnn.DataType] = None,
    layout: Optional[ttnn.Layout] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
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


def tensor_1d_to_2d_ttnn(
    tensor: "torch.Tensor",
    device: ttnn.Device,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    Convert 1D PyTorch tensor to 2D TTNN tensor without using torch.unsqueeze().

    Converts [features] -> [1, features] on device using TTNN operations.
    Used for biases, layer norm weights, and other 1D tensors.

    Args:
        tensor: 1D PyTorch tensor of shape [features]
        device: TTNN device
        dtype: TTNN data type (default: bfloat16)
        memory_config: Memory configuration (default: DRAM)

    Returns:
        TTNN tensor of shape [1, features] in TILE_LAYOUT
    """
    if dtype is None:
        dtype = ttnn.bfloat16
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Get the feature size
    features = tensor.shape[0]

    # Transfer as 1D with ROW_MAJOR (TILE requires 2D+)
    tensor_ttnn = ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    # Reshape to 2D: [features] -> [1, features]
    tensor_ttnn = ttnn.reshape(tensor_ttnn, (1, features))

    # Convert to TILE_LAYOUT
    tensor_ttnn = ttnn.to_layout(tensor_ttnn, ttnn.TILE_LAYOUT)

    return tensor_ttnn


# Default exports
create_sinusoidal_pos_embedding = create_sinusoidal_pos_embedding_ttnn
safe_cat = safe_cat_ttnn
compute_position_ids = compute_position_ids_ttnn
