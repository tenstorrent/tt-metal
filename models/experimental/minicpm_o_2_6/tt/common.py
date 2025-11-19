# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for MiniCPM-o-2_6 TTNN implementation.

Provides memory configuration helpers and tensor conversion utilities.
"""

import torch
import ttnn
from typing import Optional


def get_weights_memory_config():
    """
    Get memory configuration for weight tensors.
    Weights are stored in DRAM for large model support.

    Returns:
        ttnn.MemoryConfig: DRAM memory configuration for weights
    """
    return ttnn.DRAM_MEMORY_CONFIG


def get_activations_memory_config():
    """
    Get memory configuration for activation tensors.
    Activations are stored in L1 for faster access during computation.

    Returns:
        ttnn.MemoryConfig: L1 memory configuration for activations
    """
    return ttnn.L1_MEMORY_CONFIG


def torch_to_ttnn(
    tensor: torch.Tensor,
    device: ttnn.Device,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    dtype: Optional[ttnn.DataType] = None,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    """
    Convert PyTorch tensor to TTNN tensor.

    Args:
        tensor: PyTorch tensor to convert
        device: TTNN device to place tensor on
        memory_config: Optional memory configuration (defaults to L1 for activations)
        dtype: Optional TTNN data type (defaults to BFLOAT16)
        layout: Tensor layout (defaults to TILE_LAYOUT for optimal performance)

    Returns:
        ttnn.Tensor: Converted TTNN tensor
    """
    if memory_config is None:
        memory_config = get_activations_memory_config()

    if dtype is None:
        dtype = ttnn.bfloat16

    # Convert to TTNN tensor
    ttnn_tensor = ttnn.from_torch(
        tensor,
        device=device,
        memory_config=memory_config,
        dtype=dtype,
        layout=layout,
    )

    return ttnn_tensor


def ttnn_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    """
    Convert TTNN tensor back to PyTorch tensor.

    Args:
        tensor: TTNN tensor to convert

    Returns:
        torch.Tensor: Converted PyTorch tensor
    """
    # Convert back to PyTorch
    torch_tensor = ttnn.to_torch(tensor)

    return torch_tensor


def pad_tensor_to_tile_shape(tensor: torch.Tensor, tile_size: int = 32) -> torch.Tensor:
    """
    Pad tensor dimensions to be multiples of tile size for TTNN compatibility.

    TTNN tile layout requires dimensions to be multiples of 32.

    Args:
        tensor: Input PyTorch tensor
        tile_size: Tile size (default 32 for TTNN)

    Returns:
        torch.Tensor: Padded tensor with dimensions aligned to tile size
    """
    shape = list(tensor.shape)
    padded_shape = shape.copy()

    # Pad last two dimensions to be multiples of tile_size
    for i in [-2, -1]:
        if len(shape) >= abs(i):
            remainder = shape[i] % tile_size
            if remainder != 0:
                padded_shape[i] = shape[i] + (tile_size - remainder)

    # Apply padding if needed
    if padded_shape != shape:
        padding = []
        for orig, padded in zip(reversed(shape), reversed(padded_shape)):
            padding.extend([0, padded - orig])

        tensor = torch.nn.functional.pad(tensor, padding, mode="constant", value=0)

    return tensor


def unpad_tensor_from_tile_shape(
    tensor: torch.Tensor,
    original_shape: tuple,
) -> torch.Tensor:
    """
    Remove padding added for tile alignment.

    Args:
        tensor: Padded tensor
        original_shape: Original shape before padding

    Returns:
        torch.Tensor: Tensor with original shape restored
    """
    # Create slicing to remove padding
    slices = [slice(0, dim) for dim in original_shape]

    return tensor[slices]
