# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from ttnn import OverlappedTensor


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
            from collections import Counter

            breakdown = Counter()
            for _id, (fmt, td) in self._id_to_format.items():
                breakdown[(str(fmt), td.height, td.width)] += 1
            lines = [f"  {k}: {v}" for k, v in sorted(breakdown.items(), key=lambda x: -x[1])]
            raise RuntimeError(
                f"All {self.NUM_CIRCULAR_BUFFERS} circular buffer IDs are exhausted.\n"
                f"Trying to allocate ({data_format}, {tile.height}x{tile.width}); exclude_size={len(exclude)}\n"
                f"Breakdown:\n" + "\n".join(lines)
            )
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


def record_cb_metadata_per_coord(cb_descriptors_replicated, cb_descriptors_per_coord, cb_size_overrides=None):
    """
    Extract per-mesh-coord CB config metadata.

    Mirrors :func:`record_cb_metadata` but keyed by ``(mesh_row, mesh_col)`` —
    needed when some CBs (e.g., BSPM-SRAM weight CBs) have per-device-different
    L1 addresses.  Each coord's metadata is the merge of:
        - ``cb_descriptors_replicated``: same on every device.
        - ``cb_descriptors_per_coord[coord]``: this device's per-coord descs.

    Args:
        cb_descriptors_replicated: List of CBDescriptors identical on all devices.
        cb_descriptors_per_coord: ``dict[(mesh_row, mesh_col)] -> list[CBDescriptor]``
            holding the per-device CBDescriptors that differ across devices.
        cb_size_overrides: optional ``{cb_id: (total_size, num_pages, page_size)}``
            forced size triple recorded in place of the desc-derived values, only
            for CBs listed in ``cb_descriptors_per_coord`` (the per-coord ones).
            Real per-(device, core) addresses are still queried from each desc;
            only the size fields are uniformized.

            Use case: BSPM-SRAM produces per-(device, core)-different packed
            byte sizes, but the kernel's ``cb_wait_front(cb_in1, 1)`` requires
            ``num_pages >= 1`` and the framework's "received" counter is seeded
            from ``num_pages`` at ``setup_sharded_buffer``.  All-bfp0 cores'
            placeholder buffer (64 B < 576 B page_size) would otherwise record
            ``num_pages=0`` → ``cb_wait_front`` blocks → kernel hangs.

    Returns:
        ``dict[(mesh_row, mesh_col)] -> dict[cb_id, list[(addr, total_size,
        num_pages, page_size, core_ranges)]]`` — the per-coord equivalent of
        :func:`record_cb_metadata`'s output.
    """
    overrides = cb_size_overrides or {}
    replicated_meta = record_cb_metadata(cb_descriptors_replicated)
    out = {}
    for coord, per_coord_descs in cb_descriptors_per_coord.items():
        # Shallow-copy replicated entries — tuples are immutable so list copy
        # is enough to keep per-coord mutations isolated.
        per_coord_meta = {cb_id: list(entries) for cb_id, entries in replicated_meta.items()}
        for desc in per_coord_descs:
            for fmt in desc.format_descriptors:
                cb_id = fmt.buffer_index
                addr = ttnn.get_cb_address(desc)
                assert addr != 0, f"CB {cb_id} has address 0, which means it's not backed by a tensor"
                if cb_id in overrides:
                    total_size, num_pages, page_size = overrides[cb_id]
                else:
                    total_size = desc.total_size
                    page_size = fmt.page_size
                    num_pages = total_size // page_size
                per_coord_meta.setdefault(cb_id, []).append((addr, total_size, num_pages, page_size, desc.core_ranges))
        out[coord] = per_coord_meta
    return out


def _fill_config_block(config_block, cb_metadata, core_to_idx, _diag_label=None):
    """Write per-core CB config words into a pre-allocated ``(num_cores, WORDS_PER_CORE)`` block.

    Helper shared between the replicated and per-coord paths of
    :func:`build_cb_reconfig_tensor`.  ``config_block`` is mutated in place.
    """
    _diag_writes = []  # (cb_id, (core.x, core.y), addr) — for ordering audit
    for cb_id, entries in cb_metadata.items():
        for addr, total_size, num_pages, page_size, core_ranges in entries:
            cb_cores = ttnn.corerange_to_cores(core_ranges, row_wise=True)
            for core in cb_cores:
                key = (core.x, core.y)
                if key not in core_to_idx:
                    continue
                core_idx = core_to_idx[key]
                base = cb_id * 4
                # Detect double-write to same (cb_id, core) — would indicate
                # ordering bug where two entries claim the same CB slot on the
                # same core (e.g., replicated overwriting per-coord or vice
                # versa).  Loguru only here, in diag path.
                if config_block[core_idx, base + 0] != 0 and _diag_label is not None:
                    from loguru import logger as _diag_logger

                    _diag_logger.warning(
                        "[reconfig diag {}] double-write cb{} at core({}, {}): "
                        "prev addr=0x{:x} new addr=0x{:x} prev pages={} new pages={}",
                        _diag_label,
                        cb_id,
                        core.x,
                        core.y,
                        int(config_block[core_idx, base + 0]),
                        addr,
                        int(config_block[core_idx, base + 2]),
                        num_pages,
                    )
                config_block[core_idx, base + 0] = addr
                config_block[core_idx, base + 1] = total_size
                config_block[core_idx, base + 2] = num_pages
                config_block[core_idx, base + 3] = page_size
                if cb_id < 32:
                    config_block[core_idx, 256] |= 1 << cb_id
                else:
                    config_block[core_idx, 257] |= 1 << (cb_id - 32)
                _diag_writes.append((cb_id, key, int(addr), int(total_size), int(num_pages)))
    return _diag_writes


def build_cb_reconfig_tensor(cb_metadata=None, full_device_grid=None, mesh_device=None, *, cb_metadata_per_coord=None):
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

    Provide exactly one of ``cb_metadata`` (replicated mode — same config on
    every device, distributed via ``ReplicateTensorToMesh``) or
    ``cb_metadata_per_coord`` (per-device mode — addresses differ across the
    mesh, e.g., under BSPM SRAM; distributed via ``ShardTensor2dMesh``).

    Args:
        cb_metadata: Replicated mode — ``dict[cb_id] -> list[(addr, total_size,
            num_pages, page_size, core_ranges)]``.
        full_device_grid: CoreRangeSet covering all cores on the device.
        mesh_device: Device or MeshDevice to place the tensor on.
        cb_metadata_per_coord: Per-device mode — ``dict[(mesh_row, mesh_col)] ->
            <replicated cb_metadata shape>``; see :func:`record_cb_metadata_per_coord`.

    Returns:
        ttnn.Tensor: HEIGHT_SHARDED L1 tensor with 1 shard (264 uint32) per core.
    """
    import torch

    assert (cb_metadata is None) != (
        cb_metadata_per_coord is None
    ), "Provide exactly one of cb_metadata (replicated) or cb_metadata_per_coord (per-device)"

    all_cores = ttnn.corerange_to_cores(full_device_grid, row_wise=True)
    num_cores = len(all_cores)
    core_to_idx = {(c.x, c.y): idx for idx, c in enumerate(all_cores)}
    WORDS_PER_CORE = 264

    shard_spec = ttnn.ShardSpec(full_device_grid, (1, WORDS_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    if cb_metadata is not None:
        # Replicated path: one (num_cores, WORDS_PER_CORE) block, replicated across mesh.
        config = torch.zeros((num_cores, WORDS_PER_CORE), dtype=torch.uint32)
        _fill_config_block(config, cb_metadata, core_to_idx)

        num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if num_devices > 1 else None
        from_torch_kwargs = {"mesh_mapper": mesh_mapper} if mesh_mapper else {}
        return ttnn.from_torch(
            config,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=mem_config,
            **from_torch_kwargs,
        )

    # Per-device path: build a (mesh_rows * num_cores, mesh_cols * WORDS_PER_CORE)
    # source so ShardTensor2dMesh splits both axes — each device (r, c) ends up
    # with its own (num_cores, WORDS_PER_CORE) block.  Mirrors the layout used
    # by ShardTensor2dMesh callers elsewhere (e.g. transforms/moe.py:466).
    coords = list(cb_metadata_per_coord.keys())
    mesh_rows = max(r for r, _ in coords) + 1
    mesh_cols = max(c for _, c in coords) + 1
    source = torch.zeros((mesh_rows * num_cores, mesh_cols * WORDS_PER_CORE), dtype=torch.uint32)
    for (mr, mc), per_cb in cb_metadata_per_coord.items():
        block = source[mr * num_cores : (mr + 1) * num_cores, mc * WORDS_PER_CORE : (mc + 1) * WORDS_PER_CORE]
        _fill_config_block(block, per_cb, core_to_idx, _diag_label=f"coord({mr},{mc})")

    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(mesh_rows, mesh_cols), dims=(0, 1))
    return ttnn.from_torch(
        source,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
    )
