# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
import collections
from itertools import repeat

ALIGNMENT = 32


def _ntuple(x, n):
    if isinstance(x, collections.abc.Iterable):
        assert len(x) == n, f"{x} must be a tuple of length {n}"
        return tuple(x)
    return tuple(repeat(x, n))


def get_conv3d_config(in_channels, out_channels, kernel_size, stride, padding, padding_mode, grid_size):
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
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=T_out_block,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        output_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=1,
        compute_with_storage_grid_size=grid_size,
    )


def count_convs(obj):
    """
    Recursively count the total number of WanCausalConv3d instances in a class and its attributes.

    Args:
        obj: The object/class to search through

    Returns:
        int: Total count of WanCausalConv3d instances found
    """
    count = 0
    visited = set()

    def _count_recursive(current_obj):
        nonlocal count, visited

        # Avoid infinite recursion by tracking visited objects
        obj_id = id(current_obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        # Check if current object is a WanCausalConv3d instance
        if hasattr(current_obj, "__class__") and current_obj.__class__.__name__ == "WanCausalConv3d":
            count += 1
            return

        # Handle different types of objects
        if hasattr(current_obj, "__dict__"):
            # For objects with attributes, check each attribute
            for attr_name in dir(current_obj):
                # Skip private/magic methods and properties that might cause issues
                if attr_name.startswith("_") or attr_name in ["training", "device"]:
                    continue

                try:
                    attr_value = getattr(current_obj, attr_name)
                    # Skip methods and functions
                    if callable(attr_value) and not hasattr(attr_value, "__dict__"):
                        continue
                    _count_recursive(attr_value)
                except (AttributeError, RuntimeError, TypeError):
                    # Skip attributes that can't be accessed safely
                    continue

        elif isinstance(current_obj, (list, tuple)):
            # Handle lists and tuples
            for item in current_obj:
                _count_recursive(item)

        elif isinstance(current_obj, dict):
            # Handle dictionaries
            for value in current_obj.values():
                _count_recursive(value)

    _count_recursive(obj)
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


def prepare_conv3d_weights(mesh_device, weight, bias, conv_config, ALIGNMENT=ALIGNMENT):
    """Prepare weights and bias for TTNN."""
    C_in = weight.shape[1]
    w = weight.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out_chan
    padded_C_in = aligned_channels(C_in)
    if padded_C_in != C_in:
        w = torch.nn.functional.pad(w, (0, 0, 0, padded_C_in - C_in))

    # Reshape weights so that num_C_in_blocks is the first dimension
    kD, kH, kW, C_in_aligned, out_channels = w.shape

    C_in_block = conv_config.C_in_block
    C_in_block = C_in_aligned if C_in_block == 0 else C_in_block
    num_C_in_blocks = C_in_aligned // C_in_block
    assert (
        num_C_in_blocks * C_in_block == C_in_aligned
    ), f"num_C_in_blocks * C_in_block == C_in_aligned, got {num_C_in_blocks} * {C_in_block} != {C_in_aligned}"

    # Kernel expects num_C_in_blocks to be the first dimension to stride over it
    w = w.reshape(kD, kH, kW, num_C_in_blocks, C_in_block, out_channels)
    w = w.permute(3, 0, 1, 2, 4, 5)
    w = w.reshape(-1, out_channels)

    tt_weight = ttnn.from_torch(
        w,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
    )

    if bias is not None:
        tt_bias = ttnn.from_torch(
            bias.reshape(1, -1),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            pad_value=0,
        )
    else:
        tt_bias = None
    return tt_weight, tt_bias
