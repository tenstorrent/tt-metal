# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from models.demos.deepseek_v3_b1.blitz_overlap_tensors import OverlappedTensor


class CircularBufferIdManager:
    """Manages circular buffer ID allocation across multiple contexts.

    CB IDs can be reused across contexts when the data_format and tile match,
    but within a single context every CB ID must be unique.

    The ``tile`` argument must always be a ``ttnn.TileDescriptor`` so that
    cross-context equality comparisons are between the same object type.
    """

    NUM_CIRCULAR_BUFFERS = 64

    def __init__(self):
        self._id_to_format: dict[int, tuple] = {}
        self._next_id = 0

    def _allocate_id(self, data_format: ttnn.DataType, tile: ttnn.TileDescriptor, exclude: set[int]) -> int:
        """Find a reusable ID or allocate a new one.

        Reuse is possible when an existing ID has the same (data_format, tile)
        and is not already claimed in the caller's context (``exclude``).
        """
        if not isinstance(tile, ttnn.TileDescriptor):
            raise TypeError(f"tile must be a ttnn.TileDescriptor, got {type(tile).__name__}")
        if not isinstance(data_format, ttnn.DataType):
            raise TypeError(f"data_format must be a ttnn.DataType, got {type(data_format).__name__}")
        key = (data_format, tile)

        for cb_id, fmt_key in self._id_to_format.items():
            if fmt_key == key and cb_id not in exclude:
                return cb_id

        cb_id = self._next_id
        if cb_id >= self.NUM_CIRCULAR_BUFFERS:
            raise RuntimeError(f"All {self.NUM_CIRCULAR_BUFFERS} circular buffer IDs are exhausted")
        self._next_id += 1
        # Make a copy of the tile descriptor to avoid dependencies
        self._id_to_format[cb_id] = (data_format, ttnn.TileDescriptor(tile))
        return cb_id

    class Context:
        """A scoped view into a :class:`CircularBufferIdManager`.

        Within a single context every returned CB ID is unique.  Different
        contexts may share IDs as long as the data_format and tile match.
        """

        def __init__(self, manager: CircularBufferIdManager):
            self._manager = manager
            self._used_ids: set[int] = set()

        def get_cb_id(self, data_format: ttnn.DataType, tile: ttnn.TileDescriptor) -> int:
            cb_id = self._manager._allocate_id(data_format, tile, self._used_ids)
            self._used_ids.add(cb_id)
            return cb_id

    def create_context(self) -> "CircularBufferIdManager.Context":
        return CircularBufferIdManager.Context(self)

    def build_dummy_cb_descriptors(self, core_ranges) -> list:
        """Build minimal CB descriptors for every allocated CB ID.

        Each descriptor carries the correct format (data_format, tile, page_size)
        but uses dummy sizing (total_size = page_size) and the supplied
        ``core_ranges`` instead of real buffer addresses.  Useful when the real
        CB configuration is applied at runtime via a reconfig tensor.
        """
        descs = []
        for cb_id, (data_format, tile_desc) in self._id_to_format.items():
            # Minimal page size for dummy descriptors
            page_size = 1

            fmt = ttnn.CBFormatDescriptor(
                buffer_index=cb_id,
                data_format=data_format,
                page_size=page_size,
                tile=tile_desc,
            )
            desc = ttnn.CBDescriptor()
            desc.total_size = page_size
            desc.core_ranges = core_ranges
            desc.format_descriptors = [fmt]
            descs.append(desc)
        return descs


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


def cb_descriptor_from_overlapped_tensors(
    cb_index: int,
    overlapped_list: list[OverlappedTensor],
    fused_tensor_device: ttnn.Tensor,
    core_ranges: ttnn.CoreRangeSet = None,
) -> ttnn.CBDescriptor:
    """Create a single CBDescriptor spanning multiple OverlappedTensors in the same fused buffer.

    All tensors must share the same backing fused tensor and have identical
    dtype and tile_shape properties.  Core range sets are merged (unioned).
    """
    assert len(overlapped_list) > 0

    first = overlapped_list[0]
    core_range_override = core_ranges is not None
    if not core_range_override:
        core_ranges = first.core_range_set
    for ot in overlapped_list[1:]:
        assert ot.dtype == first.dtype
        assert ot.tile_shape == first.tile_shape
        if not core_range_override:
            core_ranges = core_ranges.merge(ot.core_range_set)

    cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
        cb_index,
        fused_tensor_device,
        core_ranges=core_ranges,
    )
    tile = ttnn.Tile(first.tile_shape)
    cb_desc.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=cb_index,
            data_format=first.dtype,
            page_size=tile.get_tile_size(first.dtype),
            tile=ttnn.TileDescriptor(tile),
        )
    ]
    return cb_desc


def record_cb_metadata(cb_descriptors):
    """
    Extract per-CB config metadata from a list of CBDescriptors.

    A single CB ID may appear more than once with different core_ranges (e.g. when
    qrope and krope share a CB ID but need different L1 addresses per core set).
    The returned dict maps cb_id → list of (addr, total_size, num_pages, page_size,
    core_ranges) so that build_cb_reconfig_tensor can write the correct config to
    each core independently.

    Args:
        cb_descriptors: List of ttnn.CBDescriptor objects

    Returns:
        dict mapping cb_id → list of (addr, total_size, num_pages, page_size, core_ranges)
    """

    cb_metadata = {}
    for desc in cb_descriptors:
        for fmt in desc.format_descriptors:
            cb_id = fmt.buffer_index
            addr = ttnn.get_cb_address(desc)
            # TODO: We should allow for non-backed CBs, and reserve their ID to prevent them from being reused.
            assert addr != 0, f"CB {cb_id} has address 0, which means it's not backed by a tensor"
            total_size = desc.total_size
            page_size = fmt.page_size
            num_pages = total_size // page_size
            cb_metadata.setdefault(cb_id, []).append((addr, total_size, num_pages, page_size, desc.core_ranges))
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
        cb_metadata: dict mapping cb_id → list of (addr, total_size, num_pages, page_size, core_ranges)
        full_device_grid: CoreRangeSet covering all cores on the device
        mesh_device: Device or MeshDevice to place the tensor on

    Returns:
        ttnn.Tensor: HEIGHT_SHARDED L1 tensor with 1 shard (264 uint32) per core
    """
    import torch

    all_cores = ttnn.corerange_to_cores(full_device_grid, row_wise=True)
    num_cores = len(all_cores)

    # Build (x, y) → index map for fast lookup
    core_to_idx = {(c.x, c.y): idx for idx, c in enumerate(all_cores)}

    # 264 words per core: 64 CBs * 4 words + 2 mask words + 6 padding
    WORDS_PER_CORE = 264
    config = torch.zeros((num_cores, WORDS_PER_CORE), dtype=torch.uint32)

    for cb_id, entries in cb_metadata.items():
        for addr, total_size, num_pages, page_size, core_ranges in entries:
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
