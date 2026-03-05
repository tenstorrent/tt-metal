# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Tensor debugging utilities for comparing tensor flows between implementations.
"""

import torch
from loguru import logger

import ttnn


def get_tensor_debug_info(tensor, name, convert_to_torch=False):
    """
    Get debug information for a tensor including shape.

    Args:
        tensor: Either a torch.Tensor or ttnn.Tensor
        name: Name/description of the tensor for logging
        convert_to_torch: If True and tensor is ttnn.Tensor, convert to torch for value checking (disabled by default)

    Returns:
        Dict with shape info
    """
    if isinstance(tensor, ttnn.Tensor):
        shape = tensor.shape
        dtype = str(tensor.dtype) if hasattr(tensor, "dtype") else "unknown"
        layout = str(tensor.layout) if hasattr(tensor, "layout") else "unknown"
        device_info = "distributed" if (hasattr(tensor, "is_distributed") and tensor.is_distributed()) else "single"
    elif torch.is_tensor(tensor):
        shape = tensor.shape
        dtype = str(tensor.dtype)
        layout = "torch"
        device_info = str(tensor.device)
    else:
        # Handle other types (e.g., tuple of tensors)
        shape = str(type(tensor))
        dtype = "unknown"
        layout = "unknown"
        device_info = "unknown"

    info = {
        "name": name,
        "shape": str(shape),
        "dtype": dtype,
        "layout": layout,
        "device": device_info,
    }

    # Log the information
    logger.info(
        f"TENSOR_DEBUG | {name} | shape={info['shape']} | dtype={info['dtype']} | layout={info['layout']} | device={info['device']}"
    )

    return info


def log_tensor_checkpoint(tensor, checkpoint_name, convert_to_torch=False):
    """
    Log tensor information at a specific checkpoint in the model flow.

    Args:
        tensor: The tensor to log
        checkpoint_name: Name of the checkpoint (e.g., "router_input", "router_output")
        convert_to_torch: If True and tensor is ttnn.Tensor, convert to torch for value checking (disabled by default)
    """
    return get_tensor_debug_info(tensor, checkpoint_name, convert_to_torch)
