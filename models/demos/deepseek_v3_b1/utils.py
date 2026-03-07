# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import struct

import ttnn


def float_to_bfloat16_packed(value):
    """Convert float to packed bfloat16 (two copies in uint32)"""
    # Convert float32 to bytes
    float_bytes = struct.pack("f", value)
    # Extract upper 16 bits (bfloat16 is truncated float32)
    bf16_bytes = float_bytes[2:4]  # upper 16 bits in little-endian layout
    # Pack two copies into uint32 (little endian)
    packed = int.from_bytes(bf16_bytes + bf16_bytes, byteorder="little")
    return packed


def float_to_uint32(value):
    """Convert float to uint32"""
    return int.from_bytes(struct.pack("f", value), byteorder="little")


def merge_per_core_runtime_args(*groups):
    """
    Merge per-core runtime arg groups in-order with core-aware concatenation.

    Each group is a list of tuples: (core_coord, list[int]).
    If a core appears in multiple groups, args are concatenated in group order.
    """
    merged = []
    core_to_index = {}
    for group in groups:
        for core, args in group:
            key = (core.x, core.y)
            args_list = list(args)
            if key in core_to_index:
                idx = core_to_index[key]
                merged_core, merged_args = merged[idx]
                merged[idx] = (merged_core, merged_args + args_list)
            else:
                core_to_index[key] = len(merged)
                merged.append((core, args_list))
    return merged


def merge_kernel_defines(*define_groups):
    """
    Merge kernel defines in-order with key-aware deduplication.

    Each input is an iterable of (name, value) tuples. First occurrence preserves
    ordering; later occurrences override the value for that define name.
    """
    merged = {}
    ordered_names = []
    for group in define_groups:
        for name, value in group:
            if name not in merged:
                ordered_names.append(name)
            merged[name] = value
    return [(name, merged[name]) for name in ordered_names]


def fabric_config_enables_torus_x(fabric_config) -> bool:
    return fabric_config in (
        ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    )


def fabric_config_enables_torus_y(fabric_config) -> bool:
    return fabric_config in (
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    )


def generate_mm_weights(shape, dtype):
    import torch

    torch_mm_weights = (torch.randn(shape, dtype=torch.float32) / (shape[-2] ** 0.5)).to(dtype)
    return torch_mm_weights
    # TODO: Review the below, which should provide a similar result
    # torch_mm_weights = torch.empty(shape, dtype=dtype)
    # # This assumes that weights are already pre-transposed, so inner dimension is the first dimension
    # # fan_in assumes the inner dimension is the second dimension, which is why we pass a transposed view
    # # Alternatively, we could pass the original shape and use fan_out
    # torch.nn.init.kaiming_normal_(torch_mm_weights.T, mode="fan_in", nonlinearity="linear")
    # return torch_mm_weights
