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
"""

from __future__ import annotations

import numpy as np
import torch

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
    ) -> None:
        assert (
            memory_config is not None and memory_config.is_sharded()
        ), "CompressedTensor requires a sharded memory_config (kernel needs contiguous tile data in L1/DRAM)"

        # Fold batch dims into height
        h = 1
        for d in tensor.shape[:-1]:
            h *= d

        # --- Tensor geometry ---
        self.shape = tensor.shape  # original ND tensor shape
        self.tile_hw = tile_hw  # tile dimension (typically 32)
        self.tiles_h = h // tile_hw  # tile rows (batch dims folded in)
        self.tiles_w = tensor.shape[-1] // tile_hw  # tile columns

        # --- Packed data on device ---
        self.data = None  # ttnn.Tensor: uint8, packed BFP tile bytes
        self.assignment = None  # ttnn.Tensor: uint8, 2-bit packed format indices
        self.spec = None  # ttnn.TensorSpec: logical shape/dtype/layout
        self.max_shard_size = 0  # bytes per shard

        # --- Assignment (host-side, for packing/unpacking) ---
        self._assignment_flat = None  # flat row-major int8 — indexed by page
        self._tile_mant_bits = []  # mant_bits per tile, in pack order (for unpack)

        # --- Shard mapping (populated by _pack_sharded) ---
        self._shard_mapping = []  # [(CoreCoord, [page_idx, ...]), ...] from C++
        self._core_assignment = {}  # {(x,y): int8[num_shard_pages]} format index per page on that core

        assert assignment.shape == (
            self.tiles_h,
            self.tiles_w,
        ), f"Assignment shape {assignment.shape} doesn't match tile grid ({self.tiles_h}, {self.tiles_w})"

        self._assignment_flat = assignment.astype(np.int8).ravel().copy()
        self._pack_data_and_assignment(tensor, memory_config, assignment_memory_config, device)

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
    ) -> CompressedTensor:
        """Convenience: run assignment then pack in one step."""
        result = assigner.assign(tensor, quantize_fn)
        return cls(
            tensor,
            result.assignment,
            device=device,
            memory_config=memory_config,
            assignment_memory_config=assignment_memory_config,
        )

    @classmethod
    def from_bspm(
        cls,
        tensor: torch.Tensor,
        assignment: np.ndarray,
        device=None,
        memory_config=None,
        assignment_memory_config=None,
        tile_hw: int = DEFAULT_TILE_HW,
    ) -> CompressedTensor:
        """Create CompressedTensor from a pre-computed assignment (e.g., BitSculpt BSPM).

        Bypasses CompressedTensorAssigner — uses the provided assignment directly.
        The assignment codes must use tt-metal's COMPRESSED_FORMATS ordering:
        0=bfp8, 1=bfp4, 2=bfp2, 3=bfp0 (use bspm_loader.py to remap from BitSculpt).

        Args:
            tensor: Original weight tensor (FP32 or BF16). Shape must be divisible
                by tile_hw in both dimensions.
            assignment: (tiles_h, tiles_w) int8/uint8 array with tt-metal format indices.
                Can be obtained from:
                - integration.ttnn.bspm_loader.load_bspm_for_expert()
                - integration.ttnn.bspm_loader.load_bspm_for_layer()["codes"][expert_idx, proj_idx]
            device: ttnn device.
            memory_config: ttnn.MemoryConfig (must be sharded).
            assignment_memory_config: Optional separate memory config for assignment tensor.
            tile_hw: Tile dimension (default 32).

        Returns:
            CompressedTensor with the BSPM-driven precision assignment.

        Example:
            from integration.ttnn.bspm_loader import load_bspm_for_expert

            assignment = load_bspm_for_expert(
                "results/deepseek-r1-0528/layer_30/precision_eval/precision_map_B_3.5.bspm",
                expert_idx=0,
                proj_idx=0,
            )
            ct = CompressedTensor.from_bspm(
                weight_tensor,
                assignment,
                device=device,
                memory_config=mem_cfg,
            )
        """
        return cls(
            tensor,
            assignment,
            device=device,
            memory_config=memory_config,
            assignment_memory_config=assignment_memory_config,
            tile_hw=tile_hw,
        )

    @classmethod
    def from_packed_data(
        cls,
        data_tensor,
        assignment_2d: np.ndarray,
        original_memory_config,
        device=None,
        tile_hw: int = DEFAULT_TILE_HW,
    ) -> CompressedTensor:
        """Reconstruct a CompressedTensor from a cached packed data tensor + assignment.

        Used when loading a previously-packed CompressedTensor from disk cache.
        The data tensor is already on device (loaded via ttnn.load_tensor); the
        assignment is a numpy array saved alongside it.  This avoids re-quantizing
        the original float weights.

        Args:
            data_tensor: Packed uint8 BFP data tensor, already on device.
            assignment_2d: (tiles_h, tiles_w) int8/uint8 array with tt-metal
                COMPRESSED_FORMATS indices.  The shape encodes the tile grid.
            original_memory_config: The MemoryConfig that was passed to __init__
                or from_bspm() during packing (before shard-size correction).
                Needed to reconstruct the shard-to-page mapping.
            device: ttnn device (required to place the repacked assignment tensor).
            tile_hw: Tile dimension (default 32).

        Returns:
            CompressedTensor ready for use with DRAMStreamingMatmulCompressed.
        """
        obj = object.__new__(cls)

        tiles_h, tiles_w = int(assignment_2d.shape[0]), int(assignment_2d.shape[1])
        K = tiles_h * tile_hw
        N_padded = tiles_w * tile_hw

        obj.shape = (K, N_padded)
        obj.tile_hw = tile_hw
        obj.tiles_h = tiles_h
        obj.tiles_w = tiles_w

        obj.data = data_tensor
        obj.spec = None
        obj.max_shard_size = data_tensor.memory_config().shard_spec.shape[1]

        obj._assignment_flat = assignment_2d.astype(np.int8).ravel().copy()
        obj._tile_mant_bits = []  # not populated; to_torch() will not work

        obj._shard_mapping = compute_shard_page_mapping((K, N_padded), original_memory_config, tile_hw)
        obj._core_assignment = obj._build_core_assignment()

        # Re-pack the assignment tensor from numpy (cheap: assignment is tiny).
        assign_bytes, assign_mem = obj._pack_sharded_assignment(
            (K, N_padded), original_memory_config, original_memory_config.buffer_type
        )
        obj.assignment = ttnn.from_torch(
            assign_bytes,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=assign_mem,
        )

        return obj

    def to_torch(self) -> torch.Tensor:
        """Unpack back to float32 torch tensor."""
        data_tensor = self.data
        if ttnn.is_tensor_storage_on_device(data_tensor):
            data_tensor = ttnn.from_device(data_tensor)
        flat_np = ttnn.to_torch(data_tensor).squeeze().numpy().astype(np.uint8)
        return self._unpack_sharded(flat_np)

    def get_assignment(self) -> np.ndarray:
        """Get assignment as (tiles_h, tiles_w) numpy array."""
        return self._assignment_flat.reshape(self.tiles_h, self.tiles_w)

    def get_assignment_per_shard(self, core_coord) -> np.ndarray:
        """Get flat assignment array for a specific core's shard."""
        assert self.max_shard_size > 0, "Per-shard assignment only for sharded tensors"
        core_key = (core_coord.x, core_coord.y)
        assert core_key in self._core_assignment, f"Core {core_coord} not found in shard grid"
        return self._core_assignment[core_key]

    def get_data_tensor(self) -> ttnn.Tensor:
        return self.data

    def get_assignment_tensor(self) -> ttnn.Tensor:
        return self.assignment

    def get_data_l1_address(self) -> int:
        assert ttnn.is_tensor_storage_on_device(self.data), "Data tensor not on device"
        return self.data.buffer_address()

    def get_assignment_l1_address(self) -> int:
        assert ttnn.is_tensor_storage_on_device(self.assignment), "Assignment tensor not on device"
        return self.assignment.buffer_address()

    @property
    def data_bytes(self) -> int:
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
        return (
            f"CompressedTensor(shape={self.shape}, tiles={self.num_tiles}, "
            f"data_bytes={self.data_bytes}{shard_str}, {fmt_str})"
        )

    # ==================================================================
    # Packing (top-level)
    # ==================================================================

    def _pack_data_and_assignment(self, tensor, memory_config, assignment_memory_config, device):
        """Pack data and assignment into sharded ttnn tensors."""
        data_bytes, self._tile_mant_bits, data_memory_config, self.max_shard_size = self._pack_sharded(
            tensor, memory_config
        )
        if assignment_memory_config is not None:
            assert (
                assignment_memory_config.is_sharded()
            ), "assignment_memory_config must be sharded when data memory_config is sharded"
            assign_buffer_type = assignment_memory_config.buffer_type
        else:
            assign_buffer_type = memory_config.buffer_type
        assign_bytes, assign_mem = self._pack_sharded_assignment(
            tensor.shape,
            memory_config,
            assign_buffer_type,
            assignment_memory_config,
        )

        logical_shape = ttnn.Shape(list(tensor.shape))
        self.spec = ttnn.TensorSpec(logical_shape, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, memory_config.buffer_type)

        self.data = ttnn.from_torch(
            data_bytes,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=data_memory_config,
        )
        self.assignment = ttnn.from_torch(
            assign_bytes,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=assign_mem,
        )

    def _pack_sharded(self, tensor, memory_config):
        """Pack tiles grouped by shard. Returns (data_torch, tile_mant_bits, mem_config, max_shard_size)."""
        data_np = self._to_2d(tensor)
        self._shard_mapping = compute_shard_page_mapping(tensor.shape, memory_config, self.tile_hw)
        self._core_assignment = self._build_core_assignment()

        shard_chunks, shard_raw_sizes, tile_mant_bits = [], [], []
        for _core, page_indices in self._shard_mapping:
            packed_list, mant_list, shard_bytes = self._pack_shard_pages(data_np, page_indices)
            shard_chunks.append(packed_list)
            shard_raw_sizes.append(shard_bytes)
            tile_mant_bits.extend(mant_list)

        max_shard_bytes = _align(max(shard_raw_sizes), _get_alignment(memory_config.buffer_type))
        data_torch = self._concat_shards_padded(shard_chunks, shard_raw_sizes, max_shard_bytes, memory_config)
        corrected_config = self._make_sharded_mem_config(memory_config, max_shard_bytes)

        return data_torch, tile_mant_bits, corrected_config, max_shard_bytes

    def _pack_sharded_assignment(self, tensor_shape, memory_config, buffer_type, assign_memory_config=None):
        """Shard the assignment tensor. Returns (assign_torch, assign_mem_config)."""
        if assign_memory_config is not None:
            shard_mapping = compute_shard_page_mapping(tensor_shape, assign_memory_config, self.tile_hw)
            mem_cfg = assign_memory_config
        else:
            shard_mapping = self._shard_mapping
            mem_cfg = memory_config

        shard_chunks, shard_raw_sizes = [], []
        for _core, page_indices in shard_mapping:
            packed = _pack_assignment(self._assignment_flat[list(page_indices)].astype(np.uint8))
            shard_chunks.append([packed])
            shard_raw_sizes.append(len(packed))

        shard_bytes = _align(max(shard_raw_sizes), _get_alignment(buffer_type))
        assign_torch = self._concat_shards_padded(shard_chunks, shard_raw_sizes, shard_bytes, mem_cfg)
        assign_config = self._make_sharded_mem_config(mem_cfg, shard_bytes)

        return assign_torch, assign_config

    # ==================================================================
    # Packing helpers
    # ==================================================================

    @staticmethod
    def _to_2d(tensor: torch.Tensor) -> np.ndarray:
        """Flatten ND tensor to 2D (all-but-last-dim folded into height)."""
        return tensor.detach().float().cpu().numpy().reshape(-1, tensor.shape[-1])

    def _pack_page(self, data_2d: np.ndarray, page_idx: int) -> tuple[np.ndarray, int]:
        """Pack a single tile by flat page index. Returns (packed_bytes, mant_bits)."""
        mant_bits = _FMT_IDX_TO_MANT_BITS[int(self._assignment_flat[page_idx])]
        if mant_bits == 0:
            return np.array([], dtype=np.uint8), mant_bits
        tr = page_idx // self.tiles_w
        tc = page_idx % self.tiles_w
        tile = data_2d[
            tr * self.tile_hw : (tr + 1) * self.tile_hw,
            tc * self.tile_hw : (tc + 1) * self.tile_hw,
        ]
        return pack_bfp_tile(tile, mant_bits), mant_bits

    def _pack_shard_pages(self, data_np, page_indices):
        """Pack all pages for one shard. Returns (packed_chunks, mant_bits_list, total_bytes)."""
        packed_list, mant_list, total_bytes = [], [], 0
        for page_idx in page_indices:
            packed, mant_bits = self._pack_page(data_np, page_idx)
            packed_list.append(packed)
            mant_list.append(mant_bits)
            total_bytes += len(packed)
        return packed_list, mant_list, total_bytes

    def _build_core_assignment(self) -> dict[tuple[int, int], np.ndarray]:
        """Build {(x, y): flat_assignment_array} from shard mapping."""
        return {
            (core.x, core.y): self._assignment_flat[list(page_indices)].astype(np.int8)
            for core, page_indices in self._shard_mapping
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
