# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CompressedTensor: a mixed-precision BFP tensor stored as raw packed bytes.

Wraps two ttnn uint8 row-major tensors:
  - data: concatenated packed BFP tiles (variable size per tile)
  - assignment: per-tile format index (uint8, one byte per tile)

The LLK kernel sees raw bytes and uses the assignment to determine
how to unpack each tile.

For sharded memory configs, tiles are grouped by shard and each shard is
padded to the same byte size so that all shards are equal (required by ttnn).
The shard-to-core mapping is derived from ttnn's BufferDistributionSpec,
guaranteeing correctness for all sharding layouts and orientations.

Multi-device (mesh) support:
  - Lockstep mode: all devices get uniform shard sizes (padded to global max
    across all devices), stored as a single mesh tensor.
  - Per-core mode: each device has different data with different compressed
    sizes per core per device, using per-device per-core tensors.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

import ttnn

from .assigner import CompressedTensorAssigner
from .tile_utils import (
    BFP_MANT_BITS,
    COMPRESSED_FORMATS,
    DEFAULT_TILE_HW,
    bfp_tile_packed_size,
    pack_bfp_tile,
    ttnn_quantize_fn,
    unpack_bfp_tile,
)

# Format index → mant_bits lookup (matches COMPRESSED_FORMATS ordering)
_FMT_IDX_TO_MANT_BITS = {idx: BFP_MANT_BITS[fmt] for idx, fmt in enumerate(COMPRESSED_FORMATS) if fmt in BFP_MANT_BITS}

_bfp_utils = ttnn._ttnn.bfp_utils

# 2-bit packing: 4 tile format indices per byte, LSB first.
_BITS_PER_TILE = 2
_TILES_PER_BYTE = 8 // _BITS_PER_TILE  # 4
_TILE_MASK = (1 << _BITS_PER_TILE) - 1  # 0x3


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_alignment(buffer_type) -> int:
    if buffer_type == ttnn.BufferType.DRAM:
        return _bfp_utils.get_dram_alignment()
    return _bfp_utils.get_l1_alignment()


def _align(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _pack_assignment(flat: np.ndarray) -> np.ndarray:
    """Pack flat uint8 assignment array (1 byte/tile) into 2-bit packed uint8 (4 tiles/byte)."""
    num_tiles = len(flat)
    num_bytes = (num_tiles + _TILES_PER_BYTE - 1) // _TILES_PER_BYTE
    packed = np.zeros(num_bytes, dtype=np.uint8)
    for i in range(num_tiles):
        packed[i // _TILES_PER_BYTE] |= (flat[i] & _TILE_MASK) << ((i % _TILES_PER_BYTE) * _BITS_PER_TILE)
    return packed


def _unpack_assignment(packed: np.ndarray, num_tiles: int) -> np.ndarray:
    """Unpack 2-bit packed uint8 array back to flat uint8 (1 byte/tile)."""
    result = np.zeros(num_tiles, dtype=np.uint8)
    for i in range(num_tiles):
        result[i] = (packed[i // _TILES_PER_BYTE] >> ((i % _TILES_PER_BYTE) * _BITS_PER_TILE)) & _TILE_MASK
    return result


def compute_shard_page_mapping(
    tensor_shape: list[int] | tuple[int, ...],
    memory_config,
    tile_hw: int = DEFAULT_TILE_HW,
) -> list[tuple]:
    """Compute shard-to-core page mapping using ttnn's BufferDistributionSpec.

    Handles all sharding layouts (HEIGHT/WIDTH/BLOCK) and orientations
    (ROW_MAJOR/COL_MAJOR) correctly via the C++ implementation.

    Args:
        tensor_shape: Original tensor shape (ND).
        memory_config: ttnn.MemoryConfig with shard spec.
        tile_hw: Tile dimension (default 32).

    Returns:
        List of (CoreCoord, [page_index, ...]) tuples — one per shard.
        Page indices are flat tile indices in row-major order.
        Order matches from_torch's shard distribution.
    """
    shard_spec = memory_config.shard_spec
    is_block = memory_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    strategy = ttnn.ShardDistributionStrategy.GRID_2D if is_block else ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
    raw = _bfp_utils.compute_shard_page_mapping(
        list(tensor_shape),
        list(shard_spec.shape),
        tile_hw,
        tile_hw,
        shard_spec.grid,
        shard_spec.orientation,
        strategy,
    )
    # C++ GRID_2D returns only cores with data (trimmed grid).
    # from_torch needs data for ALL cores in the full grid.
    # Build the full mapping: cores in from_torch order, empty pages for unused cores.
    raw_dict = {(core.x, core.y): pages for core, pages in raw}
    all_cores = ttnn.corerange_to_cores(
        shard_spec.grid,
        row_wise=(shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR),
    )
    return [(core, raw_dict.get((core.x, core.y), [])) for core in all_cores]


# ---------------------------------------------------------------------------
# CompressedTensor
# ---------------------------------------------------------------------------


class CompressedTensor:
    """A mixed-precision tensor where each 32×32 tile is independently quantized to bfp8/bfp4/bfp2.

    Packing flow (from_torch / __init__):
        1. CompressedTensorAssigner decides each tile's format (bfp8/bfp4/bfp2/bfp0)
           based on quality metrics, producing a flat assignment array.
        2. For sharded configs, compute_shard_page_mapping() calls C++ BufferDistributionSpec
           to get the canonical (CoreCoord, [page_indices]) mapping — this determines which
           tiles go to which core, matching ttnn's from_torch shard distribution exactly.
        3. Each tile is packed into variable-size BFP bytes (bfp8=1088B, bfp4=576B, bfp2=320B).
           Shards are padded to equal size and assembled into a torch tensor whose shape
           matches the sharding layout (HEIGHT/WIDTH/BLOCK).
        4. The assignment is 2-bit packed (4 tiles/byte) and sharded the same way.
        5. Both tensors are placed on device via ttnn.from_torch.

    Device-side usage:
        The LLK kernel reads the packed data bytes and uses the per-tile assignment
        to reconfigure the unpacker format on each tile. See llk_unpack_compressed.h
        and llk_custom_mm_compressed_{constexpr,runtime}.h for the
        device-side APIs.

    Unpacking flow (to_torch):
        Reverses the packing: reads packed bytes from device, iterates tiles in
        shard order (or flat order for non-sharded), unpacks each tile using its
        stored mant_bits, and places it back into a float32 output tensor.

    Multi-device support:
        When mesh_mapper is provided and device is a multi-device mesh:
        - Lockstep mode: packs each device independently, pads to global max shard
          size, creates a single mesh tensor via ShardTensorToMesh.
        - Per-core mode: packs each device independently, creates per-device per-core
          tensors via to_single_device.
    """

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(
        self,
        tensor: torch.Tensor,
        assignment: np.ndarray,
        device=None,
        memory_config=None,
        assignment_memory_config=None,
        tile_hw: int = DEFAULT_TILE_HW,
        per_core_allocation: bool = False,
        mesh_mapper_config=None,
    ) -> None:
        assert (
            memory_config is not None and memory_config.is_sharded()
        ), "CompressedTensor requires a sharded memory_config (kernel needs contiguous tile data in L1/DRAM)"

        # Fold batch dims into height
        h = 1
        for d in tensor.shape[:-1]:
            h *= d

        # --- Tensor geometry (full tensor) ---
        self.shape = tensor.shape  # original ND tensor shape
        self.tile_hw = tile_hw  # tile dimension (typically 32)
        self.tiles_h = h // tile_hw  # tile rows (batch dims folded in)
        self.tiles_w = tensor.shape[-1] // tile_hw  # tile columns

        # --- Packed data on device ---
        self.data = None  # ttnn.Tensor: uint8, packed BFP tile bytes (lockstep mode)
        self._data_per_core = {}  # {(x,y): ttnn.Tensor} per-core mode (single device)
        self.assignment = None  # ttnn.Tensor: uint8, 2-bit packed format indices (lockstep)
        self._assignment_per_core = {}  # {(x,y): ttnn.Tensor} per-core assignment tensors
        self.spec = None  # ttnn.TensorSpec: logical shape/dtype/layout
        self.max_shard_size = 0  # bytes per shard (max across cores, or global max for multi-device)

        # --- Assignment (host-side, for packing/unpacking) ---
        self._assignment_flat = None  # flat row-major int8 — indexed by page (full tensor)
        self._tile_mant_bits = []  # mant_bits per tile, in pack order (for unpack)

        # --- Shard mapping (populated by _pack_sharded) ---
        self._shard_mapping = []  # [(CoreCoord, [page_idx, ...]), ...] from C++
        self._core_assignment = {}  # {(x,y): int8[num_shard_pages]} format index per page on that core
        self._per_core_allocation = per_core_allocation

        # --- Multi-device state ---
        self._num_devices = 1
        self._mesh_mapper_config = mesh_mapper_config
        self._mesh_shape = None  # MeshShape of the mesh device
        self._per_device_shard_mapping = []  # list of per-device shard mappings
        self._per_device_tile_mant_bits = []  # list of per-device mant_bits lists
        self._per_device_core_assignment = []  # list of per-device core_assignment dicts
        self._per_device_assignment_flat = []  # list of per-device assignment arrays
        self._per_device_tiles_h = 0  # tile rows per device (same for all devices)
        self._per_device_tiles_w = 0  # tile cols per device (same for all devices)
        self._per_device_shape = ()  # tensor shape per device (same for all devices)
        # Multi-device per-core: {device_idx: {(x,y): ttnn.Tensor}}
        self._multi_device_data_per_core = {}
        self._multi_device_assignment_per_core = {}

        assert assignment.shape == (
            self.tiles_h,
            self.tiles_w,
        ), f"Assignment shape {assignment.shape} doesn't match tile grid ({self.tiles_h}, {self.tiles_w})"

        self._assignment_flat = assignment.astype(np.int8).ravel().copy()
        self._pack_data_and_assignment(
            tensor, memory_config, assignment_memory_config, device, per_core_allocation, mesh_mapper_config
        )

    # ==================================================================
    # Public API
    # ==================================================================

    @classmethod
    def from_torch(
        cls,
        tensor: torch.Tensor,
        assigner: CompressedTensorAssigner,
        device=None,
        memory_config=None,
        assignment_memory_config=None,
        quantize_fn=ttnn_quantize_fn,
        per_core_allocation: bool = True,
        mesh_mapper_config=None,
    ) -> CompressedTensor:
        """Convenience: run assignment then pack in one step."""
        result = assigner.assign(tensor, quantize_fn)
        return cls(
            tensor,
            result.assignment,
            device=device,
            memory_config=memory_config,
            assignment_memory_config=assignment_memory_config,
            per_core_allocation=per_core_allocation,
            mesh_mapper_config=mesh_mapper_config,
        )

    def to_torch(self) -> torch.Tensor:
        """Unpack back to float32 torch tensor."""
        if self._num_devices > 1:
            if self._per_core_allocation:
                return self._unpack_multi_device_per_core()
            else:
                return self._unpack_multi_device()

        if self._per_core_allocation:
            return self._unpack_per_core()
        data_tensor = self.data
        if ttnn.is_tensor_storage_on_device(data_tensor):
            data_tensor = ttnn.from_device(data_tensor)
        flat_np = ttnn.to_torch(data_tensor).squeeze().numpy().astype(np.uint8)
        return self._unpack_sharded(flat_np)

    def get_assignment(self) -> np.ndarray:
        """Get assignment as (tiles_h, tiles_w) numpy array."""
        return self._assignment_flat.reshape(self.tiles_h, self.tiles_w)

    def get_assignment_per_shard(self, core_coord, device_idx=0) -> np.ndarray:
        """Get flat assignment array for a specific core's shard."""
        if self._num_devices > 1:
            assert device_idx < len(self._per_device_core_assignment), f"device_idx {device_idx} out of range"
            core_key = (core_coord.x, core_coord.y)
            return self._per_device_core_assignment[device_idx][core_key]
        assert self.max_shard_size > 0, "Per-shard assignment only for sharded tensors"
        core_key = (core_coord.x, core_coord.y)
        assert core_key in self._core_assignment, f"Core {core_coord} not found in shard grid"
        return self._core_assignment[core_key]

    def get_data_core_range_set(self, device_idx=0):
        """Return CoreRangeSet of cores that have compressed data.

        In per_core_allocation mode, only cores with non-zero data are included.
        In lockstep mode, returns the full shard grid.
        """
        if self._num_devices > 1 and self._per_core_allocation:
            per_core = self._multi_device_data_per_core.get(device_idx, {})
            cores = [ttnn.CoreCoord(x, y) for x, y in per_core.keys()]
            return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        if self._num_devices > 1:
            shard_mapping = self._per_device_shard_mapping[device_idx]
            return ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _pages in shard_mapping])
        if self._per_core_allocation:
            cores = [ttnn.CoreCoord(x, y) for x, y in self._data_per_core.keys()]
            return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        return ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _pages in self._shard_mapping])

    def get_data_tensors(self) -> list:
        """Return data tensor(s) for use in io_tensors lists (lifetime management).

        In per_core_allocation mode, returns the list of per-core tensors.
        In lockstep mode, returns a single-element list with self.data.
        For multi-device, flattens all per-device per-core tensors.
        """
        if self._num_devices > 1 and self._per_core_allocation:
            result = []
            for dev_idx in sorted(self._multi_device_data_per_core.keys()):
                result.extend(self._multi_device_data_per_core[dev_idx].values())
            return result
        if self._per_core_allocation:
            return list(self._data_per_core.values())
        return [self.data]

    def get_assignment_tensors(self) -> list:
        """Return assignment tensor(s). Per-core list or single-element lockstep list."""
        if self._num_devices > 1 and self._per_core_allocation:
            result = []
            for dev_idx in sorted(self._multi_device_assignment_per_core.keys()):
                result.extend(self._multi_device_assignment_per_core[dev_idx].values())
            return result
        if self._assignment_per_core:
            return list(self._assignment_per_core.values())
        return [self.assignment] if self.assignment is not None else []

    def cb_descriptor_from_compressed_tensor(self, cb_index, device_idx=0):
        """Create CB descriptor(s) for the compressed data tensor.

        In per_core_allocation mode, returns a list of CBDescriptors — one per core,
        each pointing to that core's individual tensor buffer.
        In lockstep mode, returns a single-element list with the standard sharded descriptor.
        All share the same cb_index so they map to the same CB slot.
        Each CB's page_size matches its tensor's actual buffer size.
        """
        tile_32x32 = ttnn.Tile([32, 32])

        if self._num_devices > 1 and self._per_core_allocation:
            per_core = self._multi_device_data_per_core.get(device_idx, {})
            descs = []
            for core_tensor in per_core.values():
                desc = ttnn.cb_descriptor_from_sharded_tensor(cb_index, core_tensor)
                core_size = desc.total_size
                desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_index,
                        data_format=ttnn.bfloat8_b,
                        page_size=core_size,
                        tile=ttnn.TileDescriptor(tile_32x32),
                    )
                ]
                descs.append(desc)
            return descs

        if self._per_core_allocation:
            descs = []
            for core_tensor in self._data_per_core.values():
                desc = ttnn.cb_descriptor_from_sharded_tensor(cb_index, core_tensor)
                core_size = desc.total_size  # set by C++ from tensor.buffer()->aligned_size_per_bank()
                desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_index,
                        data_format=ttnn.bfloat8_b,
                        page_size=core_size,
                        tile=ttnn.TileDescriptor(tile_32x32),
                    )
                ]
                descs.append(desc)
            return descs
        else:
            desc = ttnn.cb_descriptor_from_sharded_tensor(
                cb_index,
                self.data,
                total_size=self.max_shard_size,
            )
            desc.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_index,
                    data_format=ttnn.bfloat8_b,
                    page_size=self.max_shard_size,
                    tile=ttnn.TileDescriptor(tile_32x32),
                )
            ]
            return [desc]

    def get_data_l1_address(self) -> int:
        assert (
            not self._per_core_allocation
        ), "get_data_l1_address() not available in per_core_allocation mode; use get_data_l1_address_per_core()"
        assert ttnn.is_tensor_storage_on_device(self.data), "Data tensor not on device"
        return self.data.buffer_address()

    def get_data_l1_address_per_core(self, core_coord, device_idx=0) -> int:
        """Get L1 address for a specific core's data shard.

        In per_core_allocation mode, each core has its own tensor at its own address.
        In lockstep mode, falls back to the single shared address.
        For multi-device per-core, use device_idx to select the device.
        """
        if self._num_devices > 1 and self._per_core_allocation:
            per_core = self._multi_device_data_per_core.get(device_idx, {})
            key = (core_coord.x, core_coord.y)
            assert key in per_core, f"Core {core_coord} not found in per-core data for device {device_idx}"
            return per_core[key].per_core_buffer_address(core_coord)
        if self._per_core_allocation:
            key = (core_coord.x, core_coord.y)
            assert key in self._data_per_core, f"Core {core_coord} not found in per-core data"
            return self._data_per_core[key].buffer_address()
        return self.data.buffer_address()

    def get_assignment_l1_address(self) -> int:
        """Get L1 address of assignment tensor (lockstep mode only)."""
        assert self.assignment is not None, "Assignment tensor not created (pass assignment_memory_config)"
        assert ttnn.is_tensor_storage_on_device(self.assignment), "Assignment tensor not on device"
        return self.assignment.buffer_address()

    def get_assignment_l1_address_per_core(self, core_coord, device_idx=0) -> int:
        """Get L1 address for a specific core's assignment tensor (per-core mode)."""
        if self._num_devices > 1 and self._per_core_allocation:
            per_core = self._multi_device_assignment_per_core.get(device_idx, {})
            key = (core_coord.x, core_coord.y)
            assert key in per_core, f"Core {core_coord} not found in per-core assignment for device {device_idx}"
            return per_core[key].per_core_buffer_address(core_coord)
        key = (core_coord.x, core_coord.y)
        assert key in self._assignment_per_core, f"Core {core_coord} not found in per-core assignment"
        return self._assignment_per_core[key].buffer_address()

    @property
    def data_bytes(self) -> int:
        if self._num_devices > 1:
            return sum(
                sum(bfp_tile_packed_size(mb) for mb in dev_mant_bits)
                for dev_mant_bits in self._per_device_tile_mant_bits
            )
        return sum(bfp_tile_packed_size(mb) for mb in self._tile_mant_bits)

    @property
    def num_tiles(self) -> int:
        return self.tiles_h * self.tiles_w

    @property
    def tile_counts(self) -> dict[str, int]:
        counts = {fmt: 0 for fmt in COMPRESSED_FORMATS}
        for idx in self._assignment_flat:
            counts[COMPRESSED_FORMATS[idx]] += 1
        return counts

    def __repr__(self) -> str:
        counts = self.tile_counts
        fmt_str = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
        shard_str = f", max_shard_size={self.max_shard_size}" if self.max_shard_size > 0 else ""
        dev_str = f", num_devices={self._num_devices}" if self._num_devices > 1 else ""
        return (
            f"CompressedTensor(shape={self.shape}, tiles={self.num_tiles}, "
            f"data_bytes={self.data_bytes}{shard_str}{dev_str}, {fmt_str})"
        )

    # ==================================================================
    # Packing (top-level)
    # ==================================================================

    def _pack_data_and_assignment(
        self, tensor, memory_config, assignment_memory_config, device, per_core_allocation, mesh_mapper_config
    ):
        """Pack data and assignment into sharded ttnn tensors."""
        if per_core_allocation and memory_config.buffer_type != ttnn.BufferType.L1:
            logger.warning("per_core_allocation is only supported for L1 buffers, falling back to lockstep allocation")
            per_core_allocation = False
            self._per_core_allocation = False

        num_devices = device.get_num_devices() if mesh_mapper_config is not None and device is not None else 1
        self._num_devices = num_devices
        if num_devices > 1:
            self._mesh_shape = device.shape

        if num_devices > 1:
            self._pack_multi_device(tensor, memory_config, assignment_memory_config, device, per_core_allocation)
            return

        # --- Single-device path (unchanged) ---
        data_np = self._to_2d(tensor)
        self._shard_mapping = compute_shard_page_mapping(tensor.shape, memory_config, self.tile_hw)
        self._core_assignment = self._build_core_assignment_from(self._shard_mapping, self._assignment_flat)

        # Pack each shard's tiles
        shard_chunks, shard_raw_sizes = [], []
        for _core, page_indices in self._shard_mapping:
            packed_list, mant_list, shard_bytes = self._pack_shard_pages(data_np, page_indices)
            shard_chunks.append(packed_list)
            shard_raw_sizes.append(shard_bytes)
            self._tile_mant_bits.extend(mant_list)

        alignment = _get_alignment(memory_config.buffer_type)
        self.max_shard_size = _align(max(shard_raw_sizes), alignment)

        logical_shape = ttnn.Shape(list(tensor.shape))
        self.spec = ttnn.TensorSpec(logical_shape, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, memory_config.buffer_type)

        if per_core_allocation:
            self._data_per_core = self._create_per_core_tensors(
                shard_chunks,
                shard_raw_sizes,
                _bfp_utils.get_dram_alignment(),
                memory_config,
                device,
                self._shard_mapping,
            )
        else:
            data_torch = self._concat_shards_padded(shard_chunks, shard_raw_sizes, self.max_shard_size, memory_config)
            data_memory_config = self._make_sharded_mem_config(memory_config, self.max_shard_size)
            self.data = ttnn.from_torch(
                data_torch,
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=data_memory_config,
            )

        # Assignment tensor
        if assignment_memory_config is not None:
            assert (
                assignment_memory_config.is_sharded()
            ), "assignment_memory_config must be sharded when data memory_config is sharded"
            if per_core_allocation:
                self._assignment_per_core = self._create_per_core_assignment_tensors(
                    tensor.shape,
                    memory_config,
                    assignment_memory_config,
                    device,
                    self._shard_mapping,
                    self._assignment_flat,
                )
            else:
                assign_bytes, assign_mem = self._pack_sharded_assignment(
                    tensor.shape,
                    memory_config,
                    assignment_memory_config.buffer_type,
                    assignment_memory_config,
                    self._assignment_flat,
                )
                self.assignment = ttnn.from_torch(
                    assign_bytes,
                    dtype=ttnn.uint8,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=assign_mem,
                )

    # ==================================================================
    # Multi-device packing
    # ==================================================================

    def _dev_idx_to_mesh_coord(self, dev_idx):
        """Convert flat device index (row-major) to MeshCoordinate."""
        mesh_shape = self._mesh_shape
        coords = []
        remaining = dev_idx
        for axis in reversed(range(mesh_shape.dims())):
            coords.append(remaining % mesh_shape[axis])
            remaining //= mesh_shape[axis]
        coords.reverse()
        return ttnn.MeshCoordinate(coords)

    def _split_per_device(self, tensor, device):
        """Split tensor and assignment into per-device slices based on mesh_mapper_config.

        Reads placements from the config: PlacementShard(dim) splits the tensor
        along that dim for the corresponding mesh axis; PlacementReplicate() replicates.

        The assignment is split by constructing an ND "assignment tensor" with the same
        shape as the data tensor (one element per tile row of the 2D layout, per column),
        applying the same chunk operations, and flattening back. This correctly handles
        arbitrary shard dims including middle dims of 3D+ tensors.

        Returns list of (dev_tensor, dev_assignment_flat) tuples.
        Device ordering matches ttnn's distribution order (row-major over mesh coordinates).
        """
        mesh_shape = self._mesh_shape
        placements = self._mesh_mapper_config.placements
        ndim = len(tensor.shape)
        tile_hw = self.tile_hw

        # Build an ND assignment tensor with shape matching the data tensor's tile grid.
        # The assignment is (tiles_h, tiles_w) over the 2D layout. We need to reshape it
        # to match the ND tensor structure so chunk operations preserve tile correspondence.
        #
        # ND tile shape: (shape[0]/tile_hw, shape[1]/tile_hw, ..., shape[-1]/tile_hw)
        # only works if every dim is tile-aligned. For the general case where only
        # prod(shape[:-1]) and shape[-1] are tile-aligned, we use:
        # (shape[0], shape[1], ..., shape[-2], tiles_w) where height dims are element-level
        # and width is tile-level. Then after chunking we can derive the per-device assignment.
        #
        # Simplest correct approach: build a per-element "tile index" tensor with the same
        # ND shape, chunk it identically, and extract unique tile assignments.
        # But that's memory-heavy for large tensors.
        #
        # Practical approach: chunk the DATA tensor, then for each device compute its
        # assignment by mapping per-device 2D tile positions back to full-tensor tile positions.

        # Parse placements
        axis_info = []  # [(resolved_dim | None, axis_size), ...]
        for axis in range(mesh_shape.dims()):
            axis_size = mesh_shape[axis]
            if axis < len(placements) and isinstance(placements[axis], ttnn.PlacementShard):
                dim = placements[axis].dim
                axis_info.append((dim if dim >= 0 else ndim + dim, axis_size))
            else:
                axis_info.append((None, axis_size))

        # Iterate mesh coordinates in row-major order (matching ttnn distribution)
        def _iter_coords(axis_sizes):
            if not axis_sizes:
                yield ()
                return
            for i in range(axis_sizes[0]):
                for rest in _iter_coords(axis_sizes[1:]):
                    yield (i,) + rest

        # Build a tile-index tensor: shape matches data tensor but at tile granularity.
        # tile_idx_nd[...] = flat tile index in the full 2D layout.
        # Shape: (shape[0]//tile_hw, shape[1]//tile_hw, ..., shape[-1]//tile_hw)
        # This only works if all dims are tile-aligned.
        # For the common case (2D tensors), it's always valid.
        # For ND tensors, we require all dims to be tile-aligned when using multi-device.
        nd_tile_shape = tuple(s // tile_hw for s in tensor.shape)
        total_tiles = 1
        for s in nd_tile_shape:
            total_tiles *= s
        assert total_tiles == self.tiles_h * self.tiles_w, (
            f"Multi-device requires all tensor dims to be tile-aligned (tile_hw={tile_hw}). "
            f"Shape {tensor.shape} gives ND tiles {nd_tile_shape} = {total_tiles}, "
            f"but 2D tiles = {self.tiles_h} * {self.tiles_w} = {self.tiles_h * self.tiles_w}"
        )

        # Create ND tile index tensor and ND assignment tensor
        # The flat assignment maps 2D tile (tr, tc) → format. The 2D layout is row-major
        # over the folded height. The ND tile grid folds to 2D as:
        # tile (i0, i1, ..., i_{n-2}, i_{n-1}) → 2D tile (i0 * prod(nd[1:-1]) * ... + ..., i_{n-1})
        # which is just the row-major flat index with the last dim as width.
        # So assignment_flat[flat_idx] where flat_idx is the row-major ND tile index.
        assign_nd = torch.from_numpy(self._assignment_flat.copy()).reshape(nd_tile_shape)

        result = []
        for coord in _iter_coords([s for _, s in axis_info]):
            # Chunk data tensor
            dev_t = tensor
            for axis, (shard_dim, axis_size) in enumerate(axis_info):
                if shard_dim is not None:
                    dev_t = torch.chunk(dev_t, axis_size, dim=shard_dim)[coord[axis]]

            # Chunk assignment tensor with same shard dims (at tile granularity)
            dev_assign = assign_nd
            for axis, (shard_dim, axis_size) in enumerate(axis_info):
                if shard_dim is not None:
                    dev_assign = torch.chunk(dev_assign, axis_size, dim=shard_dim)[coord[axis]]

            result.append((dev_t, dev_assign.contiguous().numpy().ravel().astype(np.int8)))

        return result

    def _pack_multi_device(self, tensor, memory_config, assignment_memory_config, device, per_core_allocation):
        """Pack data for multi-device mesh. Splits tensor per device and packs independently."""
        num_devices = self._num_devices

        # Split tensor + assignment per device
        per_device_slices = self._split_per_device(tensor, device)
        assert len(per_device_slices) == num_devices

        # All devices have the same per-device shape (even split)
        per_device_shape = per_device_slices[0][0].shape

        # Compute per-device 2D geometry
        per_device_2d_h = 1
        for d in per_device_shape[:-1]:
            per_device_2d_h *= d
        per_device_tiles_h = per_device_2d_h // self.tile_hw
        tiles_w = per_device_shape[-1] // self.tile_hw

        self._per_device_tiles_h = per_device_tiles_h
        self._per_device_tiles_w = tiles_w
        self._per_device_shape = per_device_shape

        # Pack each device independently
        all_shard_chunks = []
        all_shard_raw_sizes = []

        for dev_idx in range(num_devices):
            dev_tensor, dev_assignment = per_device_slices[dev_idx]
            dev_data_np = self._to_2d(dev_tensor)

            # Compute shard mapping for per-device shape
            shard_mapping = compute_shard_page_mapping(per_device_shape, memory_config, self.tile_hw)

            # Pack shards for this device
            shard_chunks, shard_raw_sizes = [], []
            tile_mant_bits = []
            for _core, page_indices in shard_mapping:
                packed_list, mant_list, shard_bytes = self._pack_shard_pages_ex(
                    dev_data_np, page_indices, dev_assignment, tiles_w, self.tile_hw
                )
                shard_chunks.append(packed_list)
                shard_raw_sizes.append(shard_bytes)
                tile_mant_bits.extend(mant_list)

            all_shard_chunks.append(shard_chunks)
            all_shard_raw_sizes.append(shard_raw_sizes)
            self._per_device_shard_mapping.append(shard_mapping)
            self._per_device_tile_mant_bits.append(tile_mant_bits)
            self._per_device_core_assignment.append(self._build_core_assignment_from(shard_mapping, dev_assignment))
            self._per_device_assignment_flat.append(dev_assignment)

        # Global max shard size across all devices
        alignment = _get_alignment(memory_config.buffer_type)
        global_max = max(sz for dev_sizes in all_shard_raw_sizes for sz in dev_sizes)
        self.max_shard_size = _align(global_max, alignment)

        logical_shape = ttnn.Shape(list(tensor.shape))
        self.spec = ttnn.TensorSpec(logical_shape, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, memory_config.buffer_type)

        if per_core_allocation:
            self._pack_multi_device_per_core(
                all_shard_chunks, all_shard_raw_sizes, memory_config, assignment_memory_config, device
            )
        else:
            self._pack_multi_device_lockstep(
                all_shard_chunks, all_shard_raw_sizes, memory_config, assignment_memory_config, device
            )

    def _pack_multi_device_lockstep(
        self, all_shard_chunks, all_shard_raw_sizes, memory_config, assignment_memory_config, device
    ):
        """Create lockstep mesh tensors: pad to global max, concatenate, use ShardTensorToMesh."""
        num_devices = self._num_devices
        max_shard_bytes = self.max_shard_size

        # Concatenate per-device padded data along dim 0
        per_device_torch = []
        for dev_idx in range(num_devices):
            dev_torch = self._concat_shards_padded(
                all_shard_chunks[dev_idx], all_shard_raw_sizes[dev_idx], max_shard_bytes, memory_config
            )
            per_device_torch.append(dev_torch)

        # Stack along dim 0 for ShardTensorToMesh
        combined = torch.cat(per_device_torch, dim=0)
        data_memory_config = self._make_sharded_mem_config(memory_config, max_shard_bytes)
        mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)

        self.data = ttnn.from_torch(
            combined,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=data_memory_config,
            mesh_mapper=mesh_mapper,
        )

        # Assignment tensor
        if assignment_memory_config is not None:
            assert assignment_memory_config.is_sharded(), "assignment_memory_config must be sharded"
            per_device_assign_torch = []
            for dev_idx in range(num_devices):
                dev_assign_torch, assign_mem = self._pack_sharded_assignment(
                    self._per_device_shape,
                    memory_config,
                    assignment_memory_config.buffer_type,
                    assignment_memory_config,
                    self._per_device_assignment_flat[dev_idx],
                )
                per_device_assign_torch.append(dev_assign_torch)

            combined_assign = torch.cat(per_device_assign_torch, dim=0)
            assign_mapper = ttnn.ShardTensorToMesh(device, dim=0)
            self.assignment = ttnn.from_torch(
                combined_assign,
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=assign_mem,
                mesh_mapper=assign_mapper,
            )

    def _pack_multi_device_per_core(
        self, all_shard_chunks, all_shard_raw_sizes, memory_config, assignment_memory_config, device
    ):
        """Create per-device per-core tensors using to_single_device."""
        num_devices = self._num_devices
        alignment = _bfp_utils.get_dram_alignment()

        for dev_idx in range(num_devices):
            shard_mapping = self._per_device_shard_mapping[dev_idx]
            shard_chunks = all_shard_chunks[dev_idx]
            shard_raw_sizes = all_shard_raw_sizes[dev_idx]
            coord = self._dev_idx_to_mesh_coord(dev_idx)

            per_core = {}
            for (core, _page_indices), packed_list, raw_size in zip(shard_mapping, shard_chunks, shard_raw_sizes):
                if raw_size == 0:
                    continue
                aligned_size = _align(raw_size, alignment)
                core_bytes = list(packed_list)
                pad = aligned_size - raw_size
                if pad > 0:
                    core_bytes.append(np.zeros(pad, dtype=np.uint8))
                flat = np.concatenate(core_bytes)

                core_torch = torch.from_numpy(flat.copy()).reshape(1, aligned_size)
                core_shard_spec = ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                    [1, aligned_size],
                    ttnn.ShardOrientation.ROW_MAJOR,
                )
                core_mem_config = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    memory_config.buffer_type,
                    core_shard_spec,
                    per_core_allocation=True,
                )
                host_tensor = ttnn.from_torch(core_torch, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)
                per_core[(core.x, core.y)] = ttnn._ttnn.multi_device.to_single_device(
                    host_tensor, device, coord, core_mem_config
                )

            self._multi_device_data_per_core[dev_idx] = per_core

        # Assignment per-core per-device
        if assignment_memory_config is not None:
            assert assignment_memory_config.is_sharded(), "assignment_memory_config must be sharded"
            for dev_idx in range(num_devices):
                dev_assignment = self._per_device_assignment_flat[dev_idx]
                shard_mapping = self._per_device_shard_mapping[dev_idx]
                coord = self._dev_idx_to_mesh_coord(dev_idx)

                if assignment_memory_config is not None:
                    assign_shard_mapping = compute_shard_page_mapping(
                        self._per_device_shape, assignment_memory_config, self.tile_hw
                    )
                else:
                    assign_shard_mapping = shard_mapping

                buffer_type = (
                    assignment_memory_config.buffer_type if assignment_memory_config else memory_config.buffer_type
                )
                assign_per_core = {}

                for core, page_indices in assign_shard_mapping:
                    if len(page_indices) == 0:
                        continue
                    packed = _pack_assignment(dev_assignment[list(page_indices)].astype(np.uint8))
                    raw_size = len(packed)
                    aligned_size = _align(max(raw_size, alignment), alignment)
                    pad = aligned_size - raw_size
                    if pad > 0:
                        packed = np.concatenate([packed, np.zeros(pad, dtype=np.uint8)])

                    core_torch = torch.from_numpy(packed.copy()).reshape(1, aligned_size)
                    core_shard_spec = ttnn.ShardSpec(
                        ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                        [1, aligned_size],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    )
                    core_mem_config = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        buffer_type,
                        core_shard_spec,
                        per_core_allocation=True,
                    )
                    host_tensor = ttnn.from_torch(core_torch, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)
                    assign_per_core[(core.x, core.y)] = ttnn._ttnn.multi_device.to_single_device(
                        host_tensor, device, coord, core_mem_config
                    )

                self._multi_device_assignment_per_core[dev_idx] = assign_per_core

    # ==================================================================
    # Single-device per-core tensor creation
    # ==================================================================

    @staticmethod
    def _create_per_core_tensors(shard_chunks, shard_raw_sizes, alignment, memory_config, device, shard_mapping):
        """Create one single-core ttnn tensor per core with its actual compressed size.

        Returns {(x, y): ttnn.Tensor}.
        """
        data_per_core = {}
        for (core, _page_indices), packed_list, raw_size in zip(shard_mapping, shard_chunks, shard_raw_sizes):
            if raw_size == 0:
                continue
            aligned_size = _align(raw_size, alignment)
            # Concatenate packed bytes + padding for this core
            core_bytes = list(packed_list)
            pad = aligned_size - raw_size
            if pad > 0:
                core_bytes.append(np.zeros(pad, dtype=np.uint8))
            flat = np.concatenate(core_bytes)

            core_torch = torch.from_numpy(flat.copy()).reshape(1, aligned_size)
            core_shard_spec = ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                [1, aligned_size],
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            core_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                memory_config.buffer_type,
                core_shard_spec,
                per_core_allocation=True,
            )
            data_per_core[(core.x, core.y)] = ttnn.from_torch(
                core_torch,
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=core_mem_config,
            )
        return data_per_core

    def _pack_sharded_assignment(self, tensor_shape, memory_config, buffer_type, assign_memory_config, assignment_flat):
        """Shard the assignment tensor. Returns (assign_torch, assign_mem_config)."""
        if assign_memory_config is not None:
            shard_mapping = compute_shard_page_mapping(tensor_shape, assign_memory_config, self.tile_hw)
            mem_cfg = assign_memory_config
        else:
            shard_mapping = self._shard_mapping
            mem_cfg = memory_config

        shard_chunks, shard_raw_sizes = [], []
        for _core, page_indices in shard_mapping:
            packed = _pack_assignment(assignment_flat[list(page_indices)].astype(np.uint8))
            shard_chunks.append([packed])
            shard_raw_sizes.append(len(packed))

        shard_bytes = _align(max(shard_raw_sizes), _get_alignment(buffer_type))
        assign_torch = self._concat_shards_padded(shard_chunks, shard_raw_sizes, shard_bytes, mem_cfg)
        assign_config = self._make_sharded_mem_config(mem_cfg, shard_bytes)

        return assign_torch, assign_config

    @staticmethod
    def _create_per_core_assignment_tensors(
        tensor_shape, memory_config, assign_memory_config, device, shard_mapping, assignment_flat
    ):
        """Create one single-core assignment tensor per core.

        Same pattern as _create_per_core_tensors but for the 2-bit packed assignment.
        Cores with no pages are skipped.
        Returns {(x, y): ttnn.Tensor}.
        """
        if assign_memory_config is not None:
            assign_shard_mapping = compute_shard_page_mapping(tensor_shape, assign_memory_config, DEFAULT_TILE_HW)
        else:
            assign_shard_mapping = shard_mapping

        buffer_type = assign_memory_config.buffer_type if assign_memory_config else memory_config.buffer_type
        alignment = _bfp_utils.get_dram_alignment()
        assignment_per_core = {}

        for core, page_indices in assign_shard_mapping:
            if len(page_indices) == 0:
                continue
            packed = _pack_assignment(assignment_flat[list(page_indices)].astype(np.uint8))
            raw_size = len(packed)
            aligned_size = _align(max(raw_size, alignment), alignment)
            pad = aligned_size - raw_size
            if pad > 0:
                packed = np.concatenate([packed, np.zeros(pad, dtype=np.uint8)])

            core_torch = torch.from_numpy(packed.copy()).reshape(1, aligned_size)
            core_shard_spec = ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                [1, aligned_size],
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            core_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type,
                core_shard_spec,
                per_core_allocation=True,
            )
            assignment_per_core[(core.x, core.y)] = ttnn.from_torch(
                core_torch,
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=core_mem_config,
            )
        return assignment_per_core

    # ==================================================================
    # Packing helpers
    # ==================================================================

    @staticmethod
    def _to_2d(tensor: torch.Tensor) -> np.ndarray:
        """Flatten ND tensor to 2D (all-but-last-dim folded into height)."""
        return tensor.detach().float().cpu().numpy().reshape(-1, tensor.shape[-1])

    def _pack_page(self, data_2d: np.ndarray, page_idx: int) -> tuple[np.ndarray, int]:
        """Pack a single tile by flat page index. Returns (packed_bytes, mant_bits)."""
        return self._pack_page_ex(data_2d, page_idx, self._assignment_flat, self.tiles_w, self.tile_hw)

    @staticmethod
    def _pack_page_ex(data_2d: np.ndarray, page_idx: int, assignment_flat, tiles_w, tile_hw) -> tuple[np.ndarray, int]:
        """Pack a single tile by flat page index with explicit params. Returns (packed_bytes, mant_bits)."""
        mant_bits = _FMT_IDX_TO_MANT_BITS[int(assignment_flat[page_idx])]
        if mant_bits == 0:
            return np.array([], dtype=np.uint8), mant_bits
        tr = page_idx // tiles_w
        tc = page_idx % tiles_w
        tile = data_2d[
            tr * tile_hw : (tr + 1) * tile_hw,
            tc * tile_hw : (tc + 1) * tile_hw,
        ]
        return pack_bfp_tile(tile, mant_bits), mant_bits

    def _pack_shard_pages(self, data_np, page_indices):
        """Pack all pages for one shard. Returns (packed_chunks, mant_bits_list, total_bytes)."""
        return self._pack_shard_pages_ex(data_np, page_indices, self._assignment_flat, self.tiles_w, self.tile_hw)

    @staticmethod
    def _pack_shard_pages_ex(data_np, page_indices, assignment_flat, tiles_w, tile_hw):
        """Pack all pages for one shard with explicit params. Returns (packed_chunks, mant_bits_list, total_bytes)."""
        packed_list, mant_list, total_bytes = [], [], 0
        for page_idx in page_indices:
            packed, mant_bits = CompressedTensor._pack_page_ex(data_np, page_idx, assignment_flat, tiles_w, tile_hw)
            packed_list.append(packed)
            mant_list.append(mant_bits)
            total_bytes += len(packed)
        return packed_list, mant_list, total_bytes

    def _build_core_assignment(self) -> dict[tuple[int, int], np.ndarray]:
        """Build {(x, y): flat_assignment_array} from shard mapping."""
        return self._build_core_assignment_from(self._shard_mapping, self._assignment_flat)

    @staticmethod
    def _build_core_assignment_from(shard_mapping, assignment_flat) -> dict[tuple[int, int], np.ndarray]:
        """Build {(x, y): flat_assignment_array} from shard mapping and assignment."""
        return {
            (core.x, core.y): assignment_flat[list(page_indices)].astype(np.int8)
            for core, page_indices in shard_mapping
        }

    @staticmethod
    def _concat_shards_padded(shard_chunks, shard_raw_sizes, max_shard_bytes, memory_config):
        """Concatenate per-shard packed bytes with padding into a torch tensor.

        shard_chunks has one entry per core (including empty cores).
        Tensor shape depends on layout:
          HEIGHT_SHARDED: (num_cores, shard_bytes)
          WIDTH_SHARDED:  (1, num_cores * shard_bytes)
          BLOCK_SHARDED:  (grid_h, grid_w * shard_bytes)
        """
        all_bytes = []
        for packed_list, raw_size in zip(shard_chunks, shard_raw_sizes):
            all_bytes.extend(packed_list)
            pad_size = max_shard_bytes - raw_size
            if pad_size > 0:
                all_bytes.append(np.zeros(pad_size, dtype=np.uint8))

        num_cores = len(shard_chunks)
        layout = memory_config.memory_layout
        if layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            shape = (1, num_cores * max_shard_bytes)
        elif layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            ss = memory_config.shard_spec
            grid_size = ss.grid.bounding_box().grid_size()
            is_row_major = ss.orientation == ttnn.ShardOrientation.ROW_MAJOR
            h = grid_size.y if is_row_major else grid_size.x
            w = grid_size.x if is_row_major else grid_size.y
            shape = (h, w * max_shard_bytes)
        else:  # HEIGHT_SHARDED
            shape = (num_cores, max_shard_bytes)

        flat_np = np.concatenate(all_bytes)
        return torch.from_numpy(flat_np.copy()).reshape(shape)

    @staticmethod
    def _make_sharded_mem_config(memory_config, shard_bytes):
        """Build MemoryConfig with shard shape [1, shard_bytes]."""
        ss = memory_config.shard_spec
        new_ss = ttnn.ShardSpec(ss.grid, [1, shard_bytes], ss.orientation)
        return ttnn.MemoryConfig(memory_config.memory_layout, memory_config.buffer_type, new_ss)

    # ==================================================================
    # Unpacking
    # ==================================================================

    def _unpack_per_core(self):
        """Unpack per-core tensors back to float32 torch tensor."""
        out = np.zeros((self.tiles_h * self.tile_hw, self.tiles_w * self.tile_hw), dtype=np.float32)
        mant_idx = 0
        tiles_w, tile_hw = self.tiles_w, self.tile_hw

        for core, page_indices in self._shard_mapping:
            key = (core.x, core.y)
            if key not in self._data_per_core:
                for _ in page_indices:
                    mant_idx += 1
                continue
            core_tensor = self._data_per_core[key]
            if ttnn.is_tensor_storage_on_device(core_tensor):
                core_tensor = ttnn.from_device(core_tensor)
            core_np = ttnn.to_torch(core_tensor).squeeze().numpy().astype(np.uint8)

            tile_offset = 0
            for page_idx in page_indices:
                mant_bits = self._tile_mant_bits[mant_idx]
                size = bfp_tile_packed_size(mant_bits)
                if mant_bits > 0:
                    tr, tc = page_idx // tiles_w, page_idx % tiles_w
                    tile = unpack_bfp_tile(core_np[tile_offset : tile_offset + size], mant_bits)
                    out[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw] = tile
                tile_offset += size
                mant_idx += 1

        return torch.from_numpy(out).reshape(self.shape)

    def _unpack_sharded(self, flat_np):
        """Unpack a sharded packed buffer (skip shard padding)."""
        out = np.zeros((self.tiles_h * self.tile_hw, self.tiles_w * self.tile_hw), dtype=np.float32)
        flat_np = flat_np.ravel()
        mant_idx, shard_offset = 0, 0
        tiles_w, tile_hw = self.tiles_w, self.tile_hw

        for _core, page_indices in self._shard_mapping:
            tile_offset = shard_offset
            for page_idx in page_indices:
                mant_bits = self._tile_mant_bits[mant_idx]
                size = bfp_tile_packed_size(mant_bits)
                if mant_bits > 0:
                    tr, tc = page_idx // tiles_w, page_idx % tiles_w
                    tile = unpack_bfp_tile(flat_np[tile_offset : tile_offset + size], mant_bits)
                    out[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw] = tile
                tile_offset += size
                mant_idx += 1
            shard_offset += self.max_shard_size

        return torch.from_numpy(out).reshape(self.shape)

    # ==================================================================
    # Multi-device unpacking
    # ==================================================================

    def _reconstruct_from_per_device(self, per_device_results):
        """Reconstruct the full tensor from per-device results.

        Reverses _split_per_device: concatenates along shard dims from innermost
        mesh axis outward. Devices are in row-major mesh order.
        """
        mesh_shape = self._mesh_shape
        placements = self._mesh_mapper_config.placements
        ndim = len(self._per_device_shape)

        # Parse placements (same as _split_per_device)
        axis_info = []
        for axis in range(mesh_shape.dims()):
            axis_size = mesh_shape[axis]
            if axis < len(placements) and isinstance(placements[axis], ttnn.PlacementShard):
                dim = placements[axis].dim
                axis_info.append((dim if dim >= 0 else ndim + dim, axis_size))
            else:
                axis_info.append((None, axis_size))

        # Reconstruct by concatenating from innermost axis outward
        # per_device_results is flat in row-major mesh order
        tensors = list(per_device_results)
        for axis in reversed(range(len(axis_info))):
            shard_dim, axis_size = axis_info[axis]
            if shard_dim is None:
                # Replicated: just pick one per group
                tensors = tensors[::axis_size]
            else:
                # Concatenate groups of axis_size along shard_dim
                new_tensors = []
                for i in range(0, len(tensors), axis_size):
                    new_tensors.append(torch.cat(tensors[i : i + axis_size], dim=shard_dim))
                tensors = new_tensors
        assert len(tensors) == 1
        return tensors[0]

    def _unpack_multi_device(self):
        """Unpack lockstep multi-device mesh tensor back to full float32 tensor."""
        device_tensors = ttnn.get_device_tensors(self.data)
        per_device_results = []

        for dev_idx, dev_tensor in enumerate(device_tensors):
            if ttnn.is_tensor_storage_on_device(dev_tensor):
                dev_tensor = ttnn.from_device(dev_tensor)
            flat_np = ttnn.to_torch(dev_tensor).squeeze().numpy().astype(np.uint8)

            dev_tiles_h = self._per_device_tiles_h
            dev_tiles_w = self._per_device_tiles_w
            tile_hw = self.tile_hw
            shard_mapping = self._per_device_shard_mapping[dev_idx]
            tile_mant_bits = self._per_device_tile_mant_bits[dev_idx]

            out = np.zeros((dev_tiles_h * tile_hw, dev_tiles_w * tile_hw), dtype=np.float32)
            flat_np = flat_np.ravel()
            mant_idx, shard_offset = 0, 0

            for _core, page_indices in shard_mapping:
                tile_offset = shard_offset
                for page_idx in page_indices:
                    mant_bits = tile_mant_bits[mant_idx]
                    size = bfp_tile_packed_size(mant_bits)
                    if mant_bits > 0:
                        tr, tc = page_idx // dev_tiles_w, page_idx % dev_tiles_w
                        tile = unpack_bfp_tile(flat_np[tile_offset : tile_offset + size], mant_bits)
                        out[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw] = tile
                    tile_offset += size
                    mant_idx += 1
                shard_offset += self.max_shard_size

            per_device_results.append(torch.from_numpy(out).reshape(self._per_device_shape))

        return self._reconstruct_from_per_device(per_device_results)

    def _unpack_multi_device_per_core(self):
        """Unpack per-device per-core tensors back to full float32 tensor."""
        per_device_results = []

        for dev_idx in range(self._num_devices):
            dev_tiles_h = self._per_device_tiles_h
            dev_tiles_w = self._per_device_tiles_w
            tile_hw = self.tile_hw
            shard_mapping = self._per_device_shard_mapping[dev_idx]
            tile_mant_bits = self._per_device_tile_mant_bits[dev_idx]
            data_per_core = self._multi_device_data_per_core.get(dev_idx, {})

            out = np.zeros((dev_tiles_h * tile_hw, dev_tiles_w * tile_hw), dtype=np.float32)
            mant_idx = 0

            for core, page_indices in shard_mapping:
                key = (core.x, core.y)
                if key not in data_per_core:
                    for _ in page_indices:
                        mant_idx += 1
                    continue
                core_tensor = data_per_core[key]
                if ttnn.is_tensor_storage_on_device(core_tensor):
                    core_tensor = ttnn.from_device(core_tensor)
                core_np = ttnn.to_torch(core_tensor).squeeze().numpy().astype(np.uint8)

                tile_offset = 0
                for page_idx in page_indices:
                    mant_bits = tile_mant_bits[mant_idx]
                    size = bfp_tile_packed_size(mant_bits)
                    if mant_bits > 0:
                        tr, tc = page_idx // dev_tiles_w, page_idx % dev_tiles_w
                        tile = unpack_bfp_tile(core_np[tile_offset : tile_offset + size], mant_bits)
                        out[tr * tile_hw : (tr + 1) * tile_hw, tc * tile_hw : (tc + 1) * tile_hw] = tile
                    tile_offset += size
                    mant_idx += 1

            per_device_results.append(torch.from_numpy(out).reshape(self._per_device_shape))

        return self._reconstruct_from_per_device(per_device_results)
