# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import collections
from itertools import repeat

import torch
from loguru import logger

import ttnn

from ..layers.module import Module

ALIGNMENT = 32


def _ntuple(x, n):
    if isinstance(x, collections.abc.Iterable):
        assert len(x) == n, f"{x} must be a tuple of length {n}"
        return tuple(x)
    return tuple(repeat(x, n))


def get_conv3d_config(in_channels, out_channels, kernel_size, grid_size):
    config_to_blocking = {
        # (in_channels, out_channels, kernel_size) -> (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)
        (96, 32, (3, 3, 3)): (96, 32, 1, 32, 2),
        (192, 96, (1, 3, 3)): (192, 96, 1, 4, 4),
        (96, 96, (3, 3, 3)): (96, 96, 1, 32, 2),
        (384, 192, (1, 3, 3)): (192, 96, 1, 16, 1),
        (192, 192, (3, 3, 3)): (96, 96, 1, 64, 1),
        (32, 384, (3, 3, 3)): (32, 384, 1, 8, 8),
        # (16, 384, (3, 3, 3)): (16, 32, 1, 1, 1),
        (192, 384, (3, 3, 3)): (96, 128, 1, 32, 1),
        (384, 384, (3, 3, 3)): (128, 128, 1, 16, 2),
        (384, 768, (3, 3, 3)): (128, 128, 1, 16, 2),
    }

    blocking = config_to_blocking.get((in_channels, out_channels, kernel_size), None)
    if blocking is None:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = in_channels, 32, 1, 1, 1
        logger.warning(
            f"No blocking found for {(in_channels, out_channels, kernel_size)}. Using default blocking: {C_in_block}, {C_out_block}, {T_out_block}, {H_out_block}, {W_out_block}"
        )
    else:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = blocking
    return ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=T_out_block,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=grid_size,
    )


def count_convs(module: Module) -> int:
    """
    Recursively count the total number of WanCausalConv3d instances in a class and its attributes.

    Args:
        module: The `Module` to search through

    Returns:
        int: Total count of WanCausalConv3d instances found
    """
    count = 1 if module.__class__.__name__ == "WanCausalConv3d" else 0
    for _, child in module.named_children():
        count += count_convs(child)
    return count


def conv_pad_height(tensor_BTHWC, h_factor):
    """
    For Wan2.2, in some parallism schemes height can't be fractured by the factor.
    This function pads the height to the next multiple of the factor.
    """
    B, T, H, W, C = tensor_BTHWC.shape

    # Calculate padding needed to make H divisible by h_factor
    pad_h = (h_factor - H % h_factor) % h_factor

    if pad_h > 0:
        # Pad height dimension with zeros
        tensor_BTHWC = torch.nn.functional.pad(tensor_BTHWC, (0, 0, 0, 0, 0, pad_h))

    # Return padded tensor and original height for later unpadding
    return tensor_BTHWC, H


def conv_unpad_height(tensor_BTHWC, logical_h):
    """
    For Wan2.2, remove height padding that was added by conv_pad_height.
    """
    B, T, H, W, C = tensor_BTHWC.shape
    # Slice out the original height dimension
    return tensor_BTHWC[:, :, :logical_h, :, :]


def conv_pad_in_channels(tensor):
    C_in = tensor.shape[-1]
    padded_C_in = aligned_channels(C_in)
    if padded_C_in != C_in:
        tensor = torch.nn.functional.pad(tensor, (0, padded_C_in - C_in))
    return tensor


def aligned_channels(channels):
    ALIGN_PAD = ALIGNMENT - channels % ALIGNMENT
    if channels % ALIGNMENT != 0:
        channels = channels + ALIGN_PAD
    return channels
