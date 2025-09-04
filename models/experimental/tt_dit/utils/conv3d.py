import ttnn
import torch
from loguru import logger
import collections
from itertools import repeat


def _ntuple(x, n):
    if isinstance(x, collections.abc.Iterable):
        assert len(x) == n, f"{x} must be a tuple of length {n}"
        return tuple(x)
    return tuple(repeat(x, n))


def get_conv3d_config(in_channels, out_channels, kernel_size, stride, padding, padding_mode, grid_size):
    # shape_to_blocking = {
    #     # (60, 106, 768): (128, 96, 1, 2, 16),
    #     # (120, 212, 512): (128, 128, 1, 8, 4),
    #     # (240, 424, 256): (128, 128, 4, 4, 2),
    #     # (480, 848, 128): (128, 128, 1, 2, 16),
    #     768: (128, 96, 1, 2, 16),
    #     512: (128, 128, 1, 8, 4),
    #     256: (128, 128, 4, 4, 2),
    #     128: (128, 128, 1, 2, 16),
    # }
    # blocking = shape_to_blocking.get(in_channels, None)
    blocking = None
    if blocking is None:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = in_channels, 32, 1, 1, 1
        logger.warning(
            f"No blocking found for input shape {in_channels}. Using default blocking: {C_in_block}, {C_out_block}, {T_out_block}, {H_out_block}, {W_out_block}"
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


def prepare_conv3d_weights(mesh_device, weight, bias, conv_config, ALIGNMENT=16):
    """Prepare weights and bias for TTNN."""
    C_in = weight.shape[1]
    w = weight.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out_chan
    ALIGN_PAD = ALIGNMENT - C_in % ALIGNMENT
    if C_in % ALIGNMENT != 0:
        w = torch.nn.functional.pad(w, (0, 0, 0, ALIGN_PAD))

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
