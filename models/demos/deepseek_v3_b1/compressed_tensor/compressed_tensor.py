# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CompressedTensor: a mixed-precision BFP tensor stored as raw packed bytes.

Wraps two ttnn uint8 row-major tensors:
  - data: concatenated packed BFP tiles (variable size per tile)
  - assignment: per-tile format index (uint8, one byte per tile)

The LLK kernel sees raw bytes and uses the assignment to determine
how to unpack each tile.

For sharded memory configs, tiles are grouped by shard and each shard is
padded to the same byte size so that all shards are equal (required by ttnn).
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

# L1 alignment in bytes
_bfp_utils = ttnn._ttnn.bfp_utils


def _get_alignment(buffer_type) -> int:
    """Get memory alignment in bytes for the given buffer type."""
    if buffer_type == ttnn.BufferType.DRAM:
        return _bfp_utils.get_dram_alignment()
    return _bfp_utils.get_l1_alignment()


def _align(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


# 2-bit packing: 4 tile format indices per byte, LSB first.
_BITS_PER_TILE = 2
_TILES_PER_BYTE = 8 // _BITS_PER_TILE  # 4
_TILE_MASK = (1 << _BITS_PER_TILE) - 1  # 0x3


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


class CompressedTensor:
    """A mixed-precision tensor where each 32×32 tile is independently quantized to bfp8/bfp4/bfp2.

    Attributes:
        data: ttnn uint8 row-major tensor — concatenated packed BFP tile bytes.
        assignment: ttnn uint8 row-major tensor — per-tile format index (flat, row-major).
        shape: original tensor shape (H, W).
        tiles_h: number of tile rows.
        tiles_w: number of tile columns.
        max_shard_size: bytes per shard (0 if interleaved).
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        assignment: np.ndarray,
        device=None,
        memory_config=None,
        assignment_memory_config=None,
        tile_hw: int = DEFAULT_TILE_HW,
    ) -> None:
        """Pack a float32 tensor into compressed format.

        Args:
            tensor: Float32 torch tensor, last two dims must be tile-aligned.
            assignment: (tiles_h, tiles_w) int8 array of format indices into COMPRESSED_FORMATS.
            device: Optional ttnn device. If provided, tensors are placed on device.
            memory_config: Optional ttnn.MemoryConfig for the packed data tensor.
                For sharded configs, the shard spec is recomputed to match packed tile sizes.
                If assignment_memory_config is not provided, assignment uses interleaved
                with the same buffer type.
            assignment_memory_config: Optional ttnn.MemoryConfig for the assignment tensor.
                If None, uses interleaved with same buffer type as memory_config.
            tile_hw: Tile dimension (default 32).
        """
        self.shape = tensor.shape
        self.tile_hw = tile_hw
        # Fold batch dims into height: (B1, B2, ..., H, W) → tiles_h covers all dims except W
        h = 1
        for d in tensor.shape[:-1]:
            h *= d
        self.tiles_h = h // tile_hw
        self.tiles_w = tensor.shape[-1] // tile_hw

        assert assignment.shape == (
            self.tiles_h,
            self.tiles_w,
        ), f"Assignment shape {assignment.shape} doesn't match tile grid ({self.tiles_h}, {self.tiles_w})"

        is_sharded = memory_config is not None and memory_config.is_sharded()

        if is_sharded:
            data_bytes, self._tile_mant_bits, data_memory_config, self.max_shard_size = self._pack_sharded(
                tensor, assignment, memory_config
            )
            # Assignment must shard on the same grid/layout as data so each core
            # gets matching tiles. User can override buffer_type via assignment_memory_config.
            if assignment_memory_config is not None:
                assert (
                    assignment_memory_config.is_sharded()
                ), "assignment_memory_config must be sharded when data memory_config is sharded"
                assert assignment_memory_config.memory_layout == memory_config.memory_layout, (
                    f"Assignment shard layout {assignment_memory_config.memory_layout} must match "
                    f"data shard layout {memory_config.memory_layout}"
                )
                a_grid = assignment_memory_config.shard_spec.grid
                d_grid = memory_config.shard_spec.grid
                assert a_grid == d_grid, f"Assignment shard grid {a_grid} must match data shard grid {d_grid}"
                assert (
                    assignment_memory_config.shard_spec.orientation == memory_config.shard_spec.orientation
                ), "Assignment shard orientation must match data shard orientation"
                assign_buffer_type = assignment_memory_config.buffer_type
            else:
                assign_buffer_type = memory_config.buffer_type

            assign_bytes, assign_memory_config = self._shard_assignment(assignment, memory_config, assign_buffer_type)
        else:
            data_bytes, self._tile_mant_bits = self._pack(tensor, assignment)
            data_memory_config = memory_config
            self.max_shard_size = 0
            packed_assign = _pack_assignment(assignment.astype(np.uint8).ravel())
            assign_bytes = torch.from_numpy(packed_assign).unsqueeze(0)
            assign_memory_config = assignment_memory_config or memory_config

        # Store the original tensor's spec (shape + layout + memory config).
        # Preserves the logical view for downstream use (e.g., shard mapping, reshape).
        # For sharded configs, we compute the correct shard spec for the original float32 shape
        # (not the user's dummy shard spec which is for the packed byte layout).
        logical_shape = ttnn.Shape(list(tensor.shape))
        buffer_type = memory_config.buffer_type if memory_config is not None else ttnn.BufferType.DRAM
        self.spec = ttnn.TensorSpec(
            logical_shape,
            ttnn.float32,
            ttnn.ROW_MAJOR_LAYOUT,
            buffer_type,
        )

        # Store packed data as ttnn uint8 row-major tensors
        self.data = ttnn.from_torch(
            data_bytes if is_sharded else data_bytes.unsqueeze(0),
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
            memory_config=assign_memory_config,
        )

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

    def to_torch(self) -> torch.Tensor:
        """Unpack back to float32 torch tensor."""
        data_tensor = self.data
        if ttnn.is_tensor_storage_on_device(data_tensor):
            data_tensor = ttnn.from_device(data_tensor)
        flat_np = ttnn.to_torch(data_tensor).squeeze().numpy().astype(np.uint8)

        if self.max_shard_size > 0:
            return self._unpack_sharded(flat_np)
        return self._unpack_flat(flat_np)

    def get_assignment_numpy(self) -> np.ndarray:
        """Get assignment as (tiles_h, tiles_w) numpy array. Unpacks 2-bit encoding."""
        assign_tensor = self.assignment
        if ttnn.is_tensor_storage_on_device(assign_tensor):
            assign_tensor = ttnn.from_device(assign_tensor)
        raw = ttnn.to_torch(assign_tensor).numpy().astype(np.uint8)

        if self.max_shard_size > 0:
            # Sharded: flatten and split by shard byte size, unpack each shard
            flat = raw.ravel()
            num_shards = len(self._shard_tile_coords)
            shard_bytes = len(flat) // num_shards
            result = np.zeros(self.tiles_h * self.tiles_w, dtype=np.int8)
            for shard_idx, tile_coords in enumerate(self._shard_tile_coords):
                shard_start = shard_idx * shard_bytes
                shard_packed = flat[shard_start : shard_start + shard_bytes]
                shard_unpacked = _unpack_assignment(shard_packed, len(tile_coords))
                for i, (tr, tc) in enumerate(tile_coords):
                    result[tr * self.tiles_w + tc] = shard_unpacked[i]
            return result.reshape(self.tiles_h, self.tiles_w)
        else:
            num_tiles = self.tiles_h * self.tiles_w
            return _unpack_assignment(raw.ravel(), num_tiles).astype(np.int8).reshape(self.tiles_h, self.tiles_w)

    @property
    def data_bytes(self) -> int:
        """Total packed data size in bytes (excluding shard padding)."""
        return sum(bfp_tile_packed_size(mb) for mb in self._tile_mant_bits)

    @property
    def num_tiles(self) -> int:
        return self.tiles_h * self.tiles_w

    def get_data_tensor(self) -> ttnn.Tensor:
        """Get the packed data ttnn tensor (uint8, row-major)."""
        return self.data

    def get_assignment_tensor(self) -> ttnn.Tensor:
        """Get the assignment ttnn tensor (uint8, 2-bit packed, row-major)."""
        return self.assignment

    def get_data_l1_address(self) -> int:
        """Get the L1 address of the packed data tensor. Must be on device."""
        assert ttnn.is_tensor_storage_on_device(self.data), "Data tensor not on device"
        return self.data.buffer_address()

    def get_assignment_l1_address(self) -> int:
        """Get the L1 address of the assignment tensor. Must be on device."""
        assert ttnn.is_tensor_storage_on_device(self.assignment), "Assignment tensor not on device"
        return self.assignment.buffer_address()

    @property
    def tile_counts(self) -> dict[str, int]:
        """Count of tiles per format."""
        assign = self.get_assignment_numpy().ravel()
        counts = {fmt: 0 for fmt in COMPRESSED_FORMATS}
        for idx in assign:
            counts[COMPRESSED_FORMATS[idx]] += 1
        return counts

    # ------------------------------------------------------------------
    # Internal packing
    # ------------------------------------------------------------------

    def _to_2d(self, tensor: torch.Tensor) -> np.ndarray:
        """Flatten ND tensor to 2D (all-but-last-dim folded into height)."""
        return tensor.detach().float().cpu().numpy().reshape(-1, tensor.shape[-1])

    def _pack_tile(self, data_2d: np.ndarray, tr: int, tc: int, assignment: np.ndarray) -> tuple[np.ndarray, int]:
        """Pack a single tile from 2D data, return (packed_bytes, mant_bits)."""
        mant_bits = _FMT_IDX_TO_MANT_BITS[int(assignment[tr, tc])]
        if mant_bits == 0:
            # fp0: no data stored
            return np.array([], dtype=np.uint8), mant_bits
        tile = data_2d[
            tr * self.tile_hw : (tr + 1) * self.tile_hw,
            tc * self.tile_hw : (tc + 1) * self.tile_hw,
        ]
        return pack_bfp_tile(tile, mant_bits), mant_bits

    def _pack(self, tensor: torch.Tensor, assignment: np.ndarray) -> tuple[torch.Tensor, list[int]]:
        """Pack tiles into flat uint8 buffer (no shard padding)."""
        data_np = self._to_2d(tensor)
        chunks = []
        tile_mant_bits = []
        for tr in range(self.tiles_h):
            for tc in range(self.tiles_w):
                packed, mant_bits = self._pack_tile(data_np, tr, tc, assignment)
                tile_mant_bits.append(mant_bits)
                chunks.append(packed)
        flat_np = np.concatenate(chunks)
        return torch.from_numpy(flat_np.copy()), tile_mant_bits

    def _pack_sharded(
        self, tensor: torch.Tensor, assignment: np.ndarray, memory_config
    ) -> tuple[torch.Tensor, list[int], ttnn.MemoryConfig, int]:
        """Pack tiles grouped by shard, with per-shard padding.

        Groups tiles by shard based on the memory config layout, packs each shard,
        pads all shards to the same byte size, then reshapes to (num_shards, max_shard_bytes)
        and lets from_torch handle the rest.

        Returns (data_bytes, tile_mant_bits, corrected_memory_config, max_shard_size).
        """
        data_np = self._to_2d(tensor)
        layout = memory_config.memory_layout
        shard_spec = memory_config.shard_spec
        num_cores = shard_spec.num_cores()

        # Use TensorSpec sharding helpers to get the canonical shard shape (in elements),
        # then derive tile-to-core mapping from that. This guarantees we match ttnn's distribution.
        shard_tile_coords = self._compute_shard_tile_mapping(memory_config)

        # Pack each shard's tiles and find max shard size
        shard_chunks = []
        shard_raw_sizes = []
        for tile_coords in shard_tile_coords:
            packed_list = []
            mant_list = []
            shard_byte_count = 0
            for tr, tc in tile_coords:
                packed, mant_bits = self._pack_tile(data_np, tr, tc, assignment)
                packed_list.append(packed)
                mant_list.append(mant_bits)
                shard_byte_count += len(packed)
            shard_chunks.append((packed_list, mant_list))
            shard_raw_sizes.append(shard_byte_count)

        alignment = _get_alignment(memory_config.buffer_type)
        max_shard_bytes = _align(max(shard_raw_sizes), alignment)
        self._shard_tile_coords = shard_tile_coords

        # Concatenate shards with padding, each shard is max_shard_bytes
        all_bytes = []
        tile_mant_bits = []
        for (packed_list, mant_list), raw_size in zip(shard_chunks, shard_raw_sizes):
            for packed in packed_list:
                all_bytes.append(packed)
            tile_mant_bits.extend(mant_list)
            pad_size = max_shard_bytes - raw_size
            if pad_size > 0:
                all_bytes.append(np.zeros(pad_size, dtype=np.uint8))

        flat_np = np.concatenate(all_bytes)

        grid_size = shard_spec.grid.bounding_box().grid_size()
        grid_h, grid_w = grid_size.y, grid_size.x
        tensor_shape, shard_shape = self._sharded_tensor_shape(
            layout, num_cores, max_shard_bytes, grid_h, grid_w, shard_spec.orientation
        )
        data_torch = torch.from_numpy(flat_np.copy()).reshape(tensor_shape)
        corrected_shard_spec = ttnn.ShardSpec(shard_spec.grid, shard_shape, shard_spec.orientation)
        corrected_config = ttnn.MemoryConfig(layout, memory_config.buffer_type, corrected_shard_spec)

        return data_torch, tile_mant_bits, corrected_config, max_shard_bytes

    def _shard_assignment(
        self, assignment: np.ndarray, memory_config, buffer_type
    ) -> tuple[torch.Tensor, ttnn.MemoryConfig]:
        """Shard the assignment tensor to match data sharding.

        Same grid and layout as data so each core's assignment aligns with its tiles.
        Each shard's byte count is aligned to _MAX_MEMORY_ALIGNMENT. The last shard
        may have fewer tiles (ttnn handles partial shards natively).

        Returns (assign_torch, assign_memory_config).
        """
        shard_spec = memory_config.shard_spec
        num_cores = shard_spec.num_cores()

        # Build per-shard assignment bytes (1 byte per tile)
        assign_shards = []
        for tile_coords in self._shard_tile_coords:
            shard_assign = np.array([assignment[tr, tc] for tr, tc in tile_coords], dtype=np.uint8)
            assign_shards.append(shard_assign)

        # Pack each shard's assignments to 2-bit, then pad to alignment
        max_tiles = max(len(s) for s in assign_shards)
        packed_shard_bytes = (max_tiles + _TILES_PER_BYTE - 1) // _TILES_PER_BYTE
        alignment = _get_alignment(buffer_type)
        shard_bytes = _align(packed_shard_bytes, alignment)

        padded_shards = []
        for shard in assign_shards:
            packed = _pack_assignment(shard)
            pad_size = shard_bytes - len(packed)
            if pad_size > 0:
                packed = np.concatenate([packed, np.zeros(pad_size, dtype=np.uint8)])
            padded_shards.append(packed)

        flat_np = np.concatenate(padded_shards)
        layout = memory_config.memory_layout
        grid_size = shard_spec.grid.bounding_box().grid_size()
        grid_h, grid_w = grid_size.y, grid_size.x

        assign_shape, assign_shard_shape = self._sharded_tensor_shape(
            layout, num_cores, shard_bytes, grid_h, grid_w, shard_spec.orientation
        )
        assign_torch = torch.from_numpy(flat_np.copy()).reshape(assign_shape)
        assign_shard_spec_new = ttnn.ShardSpec(shard_spec.grid, assign_shard_shape, shard_spec.orientation)
        assign_config = ttnn.MemoryConfig(layout, buffer_type, assign_shard_spec_new)

        return assign_torch, assign_config

    @staticmethod
    def _sharded_tensor_shape(
        layout, num_cores: int, shard_bytes: int, grid_h: int, grid_w: int, orientation
    ) -> tuple[tuple[int, int], list[int]]:
        """Compute tensor shape and shard shape for a given sharding layout.

        Returns (tensor_shape, shard_shape) where:
          HEIGHT_SHARDED: (num_cores * shard_bytes, 1), shard = [shard_bytes, 1]
          WIDTH_SHARDED:  (1, num_cores * shard_bytes), shard = [1, shard_bytes]
          BLOCK_SHARDED:  (h_shards * shard_bytes, w_shards), shard = [shard_bytes, 1]
        """
        if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            return (num_cores * shard_bytes, 1), [shard_bytes, 1]
        elif layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            return (1, num_cores * shard_bytes), [1, shard_bytes]
        elif layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            is_row_major = orientation == ttnn.ShardOrientation.ROW_MAJOR
            h_shards = grid_h if is_row_major else grid_w
            w_shards = grid_w if is_row_major else grid_h
            return (h_shards * shard_bytes, w_shards), [shard_bytes, 1]
        else:
            raise ValueError(f"Unsupported sharded layout: {layout}")

    def _compute_shard_tile_mapping(self, memory_config) -> list[list[tuple[int, int]]]:
        """Compute which tiles go to each shard from the memory config's shard shape.

        Uses the shard shape from the user's memory config directly (converted to tiles).
        The user is responsible for providing a valid shard spec (e.g., from TensorSpec
        sharding helpers). This guarantees our tile distribution matches ttnn exactly.

        Returns list of num_cores lists, each containing (tr, tc) tile coordinates.
        """
        layout = memory_config.memory_layout
        shard_spec = memory_config.shard_spec
        grid = shard_spec.grid
        orientation = shard_spec.orientation
        tile_hw = self.tile_hw
        num_cores = grid.num_cores()

        # Convert shard shape from elements to tiles
        shard_h_tiles = shard_spec.shape[0] // tile_hw
        shard_w_tiles = shard_spec.shape[1] // tile_hw

        if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            shards = []
            row = 0
            for _ in range(num_cores):
                row_count = min(shard_h_tiles, self.tiles_h - row)
                row_count = max(row_count, 0)
                coords = [(tr, tc) for tr in range(row, row + row_count) for tc in range(self.tiles_w)]
                shards.append(coords)
                row += row_count
            return shards

        elif layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            shards = []
            col = 0
            for _ in range(num_cores):
                col_count = min(shard_w_tiles, self.tiles_w - col)
                col_count = max(col_count, 0)
                coords = [(tr, tc) for tr in range(self.tiles_h) for tc in range(col, col + col_count)]
                shards.append(coords)
                col += col_count
            return shards

        elif layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            assert len(grid.ranges()) == 1, "Block sharding requires a single contiguous rectangular grid"
            grid_size = grid.bounding_box().grid_size()
            grid_h, grid_w = grid_size.y, grid_size.x
            is_row_major = orientation == ttnn.ShardOrientation.ROW_MAJOR
            h_shards = grid_h if is_row_major else grid_w
            w_shards = grid_w if is_row_major else grid_h

            shards = []
            row = 0
            for _ in range(h_shards):
                row_count = min(shard_h_tiles, self.tiles_h - row)
                row_count = max(row_count, 0)
                col = 0
                for _ in range(w_shards):
                    col_count = min(shard_w_tiles, self.tiles_w - col)
                    col_count = max(col_count, 0)
                    coords = [(tr, tc) for tr in range(row, row + row_count) for tc in range(col, col + col_count)]
                    shards.append(coords)
                    col += col_count
                row += row_count
            return shards

        else:
            raise ValueError(f"Unsupported sharded layout: {layout}")

    # ------------------------------------------------------------------
    # Internal unpacking
    # ------------------------------------------------------------------

    def _unpack_flat(self, flat_np: np.ndarray) -> torch.Tensor:
        """Unpack a flat (non-sharded) packed buffer."""
        out = np.zeros((self.tiles_h * self.tile_hw, self.tiles_w * self.tile_hw), dtype=np.float32)
        offset = 0
        idx = 0
        for tr in range(self.tiles_h):
            for tc in range(self.tiles_w):
                mant_bits = self._tile_mant_bits[idx]
                size = bfp_tile_packed_size(mant_bits)
                if mant_bits > 0:
                    tile = unpack_bfp_tile(flat_np[offset : offset + size], mant_bits)
                    out[tr * self.tile_hw : (tr + 1) * self.tile_hw, tc * self.tile_hw : (tc + 1) * self.tile_hw] = tile
                # fp0: out is already zeros
                offset += size
                idx += 1
        return torch.from_numpy(out).reshape(self.shape)

    def _unpack_sharded(self, flat_np: np.ndarray) -> torch.Tensor:
        """Unpack a sharded packed buffer (skip shard padding)."""
        out = np.zeros((self.tiles_h * self.tile_hw, self.tiles_w * self.tile_hw), dtype=np.float32)
        flat_np = flat_np.ravel()
        mant_idx = 0
        shard_offset = 0

        for shard_tiles in self._shard_tile_coords:
            tile_offset = shard_offset
            for tr, tc in shard_tiles:
                mant_bits = self._tile_mant_bits[mant_idx]
                size = bfp_tile_packed_size(mant_bits)
                if mant_bits > 0:
                    tile = unpack_bfp_tile(flat_np[tile_offset : tile_offset + size], mant_bits)
                    out[tr * self.tile_hw : (tr + 1) * self.tile_hw, tc * self.tile_hw : (tc + 1) * self.tile_hw] = tile
                # fp0: out is already zeros
                tile_offset += size
                mant_idx += 1
            # Skip to next shard (past padding)
            shard_offset += self.max_shard_size

        return torch.from_numpy(out).reshape(self.shape)

    def __repr__(self) -> str:
        counts = self.tile_counts
        fmt_str = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
        shard_str = f", max_shard_size={self.max_shard_size}" if self.max_shard_size > 0 else ""
        return (
            f"CompressedTensor(shape={self.shape}, tiles={self.num_tiles}, "
            f"data_bytes={self.data_bytes}{shard_str}, {fmt_str})"
        )
