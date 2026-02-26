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


def record_cb_metadata(cb_descriptors):
    """
    Extract per-CB config metadata from a list of CBDescriptors.

    Args:
        cb_descriptors: List of ttnn.CBDescriptor objects

    Returns:
        dict mapping cb_id → (addr, total_size, num_pages, page_size, core_ranges)
    """
    import ttnn

    cb_metadata = {}
    for desc in cb_descriptors:
        for fmt in desc.format_descriptors:
            cb_id = fmt.buffer_index
            addr = ttnn.get_cb_address(desc)
            total_size = desc.total_size
            page_size = fmt.page_size
            num_pages = total_size // page_size
            cb_metadata[cb_id] = (addr, total_size, num_pages, page_size, desc.core_ranges)
    return cb_metadata


def build_cb_reconfig_tensor(cb_metadata, full_device_grid, mesh_device):
    """
    Build an L1-sharded tensor containing per-core CB config data for reconfig.

    When fusing kernels, the preceding op leaves its own CB configuration in firmware.
    This function creates a tensor that the fused kernel reads at startup to reconfigure
    its CB read/write interfaces via setup_local_cb_read_write_interfaces().

    Per-core layout (264 uint32 = 1056 bytes, 32B-aligned):
        Words 0-255:   64 CB configs, 4 words each:
                       [addr_bytes, size_bytes, num_pages, page_size_bytes]
                       Zeros for unused CBs.
        Words 256:     cb_mask_low  (bits 0-31)
        Words 257:     cb_mask_high (bits 32-63)
        Words 258-259: cross-RISC sync semaphores (initialized to 0, used at runtime)
        Words 260-263: reserved (zeros)

    Args:
        cb_metadata: dict mapping cb_id → (addr, total_size, num_pages, page_size, core_ranges)
        full_device_grid: CoreRangeSet covering all cores on the device
        mesh_device: Device or MeshDevice to place the tensor on

    Returns:
        ttnn.Tensor: HEIGHT_SHARDED L1 tensor with 1 shard (264 uint32) per core
    """
    import torch

    import ttnn

    all_cores = ttnn.corerange_to_cores(full_device_grid, row_wise=True)
    num_cores = len(all_cores)

    # Build (x, y) → index map for fast lookup
    core_to_idx = {(c.x, c.y): idx for idx, c in enumerate(all_cores)}

    # 264 words per core: 64 CBs * 4 words + 2 mask words + 6 padding
    WORDS_PER_CORE = 264
    config = torch.zeros((num_cores, WORDS_PER_CORE), dtype=torch.uint32)

    for cb_id, (addr, total_size, num_pages, page_size, core_ranges) in cb_metadata.items():
        cb_cores = ttnn.corerange_to_cores(core_ranges, row_wise=True)
        for core in cb_cores:
            key = (core.x, core.y)
            if key not in core_to_idx:
                continue
            core_idx = core_to_idx[key]
            base = cb_id * 4
            config[core_idx, base + 0] = addr
            config[core_idx, base + 1] = total_size
            config[core_idx, base + 2] = num_pages
            config[core_idx, base + 3] = page_size
            if cb_id < 32:
                config[core_idx, 256] |= 1 << cb_id
            else:
                config[core_idx, 257] |= 1 << (cb_id - 32)

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if num_devices > 1 else None
    from_torch_kwargs = {"mesh_mapper": mesh_mapper} if mesh_mapper else {}

    shard_spec = ttnn.ShardSpec(full_device_grid, (1, WORDS_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    return ttnn.from_torch(
        config,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=mem_config,
        **from_torch_kwargs,
    )
