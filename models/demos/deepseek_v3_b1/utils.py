# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from models.demos.deepseek_v3_b1.blitz_decode_weights import OverlappedTensor


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


def cb_descriptor_from_overlapped_tensor(
    cb_index: int,
    overlapped: OverlappedTensor,
    fused_tensor_device: ttnn.Tensor,
) -> ttnn.CBDescriptor:
    """Create a CBDescriptor from an OverlappedTensor view backed by a fused device tensor.

    Uses ``cb_descriptor_from_sharded_tensor`` for buffer/address plumbing,
    then replaces the format descriptor so that tile shape, page size, and
    data format all reflect the *sub-tensor's* properties (which may differ
    from the fused container).  The ``CBFormatDescriptor`` constructor
    accepting ``ttnn.DataType`` is used so the DataType→DataFormat
    conversion happens automatically in C++.
    """
    cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
        cb_index,
        fused_tensor_device,
        address_offset=overlapped.byte_offset,
        total_size=overlapped.total_size,
        core_ranges=overlapped.core_range_set,
    )
    tile = ttnn.Tile(overlapped.tile_shape)
    cb_desc.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=cb_index,
            data_format=overlapped.dtype,
            page_size=tile.get_tile_size(overlapped.dtype),
            tile=ttnn.TileDescriptor(tile),
        )
    ]
    return cb_desc
