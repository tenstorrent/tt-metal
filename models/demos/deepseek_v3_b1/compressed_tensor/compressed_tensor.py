# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CompressedTensor: a mixed-precision BFP tensor stored as raw packed bytes.

Wraps two ttnn uint8 row-major tensors:
  - data: concatenated packed BFP tiles (variable size per tile)
  - assignment: per-tile format index (uint8, one byte per tile)

The LLK kernel sees raw bytes and uses the assignment to determine
how to unpack each tile.
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn

from .assigner import CompressedTensorAssigner
from .tile_utils import (
    BFP_MANT_BITS,
    COMPRESSED_FORMATS,
    bfp_tile_packed_size,
    pack_bfp_tile,
    ttnn_quantize_fn,
    unpack_bfp_tile,
)

# Format index → mant_bits lookup (matches COMPRESSED_FORMATS ordering)
_FMT_IDX_TO_MANT_BITS = {idx: BFP_MANT_BITS[fmt] for idx, fmt in enumerate(COMPRESSED_FORMATS) if fmt in BFP_MANT_BITS}


class CompressedTensor:
    """A mixed-precision tensor where each 32×32 tile is independently quantized to bfp8/bfp4/bfp2.

    Attributes:
        data: ttnn uint8 row-major tensor — concatenated packed BFP tile bytes.
        assignment: ttnn uint8 row-major tensor — per-tile format index (flat, row-major).
        shape: original tensor shape (H, W).
        tiles_h: number of tile rows.
        tiles_w: number of tile columns.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        assignment: np.ndarray,
        device=None,
        tile_hw: int = 32,
    ) -> None:
        """Pack a float32 tensor into compressed format.

        Args:
            tensor: Float32 torch tensor, last two dims must be tile-aligned.
            assignment: (tiles_h, tiles_w) int8 array of format indices into COMPRESSED_FORMATS.
            device: Optional ttnn device. If provided, tensors are placed on device.
            tile_hw: Tile dimension (default 32).
        """
        self.shape = tensor.shape
        self.tile_hw = tile_hw
        self.tiles_h = tensor.shape[-2] // tile_hw
        self.tiles_w = tensor.shape[-1] // tile_hw

        assert assignment.shape == (
            self.tiles_h,
            self.tiles_w,
        ), f"Assignment shape {assignment.shape} doesn't match tile grid ({self.tiles_h}, {self.tiles_w})"

        # Pack tiles
        data_bytes, self._tile_mant_bits = self._pack(tensor, assignment)

        # Store as ttnn uint8 row-major tensors
        self.data = ttnn.from_torch(
            data_bytes.unsqueeze(0), dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        assign_flat = torch.from_numpy(assignment.astype(np.uint8).ravel())
        self.assignment = ttnn.from_torch(
            assign_flat.unsqueeze(0), dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        assigner: CompressedTensorAssigner,
        device=None,
        quantize_fn=ttnn_quantize_fn,
    ) -> CompressedTensor:
        """Convenience: run assignment then pack in one step."""
        result = assigner.assign(tensor, quantize_fn)
        return cls(tensor, result.assignment, device=device)

    def unpack(self) -> torch.Tensor:
        """Unpack back to float32 torch tensor."""
        data_tensor = self.data
        if ttnn.is_tensor_storage_on_device(data_tensor):
            data_tensor = ttnn.from_device(data_tensor)
        flat_np = ttnn.to_torch(data_tensor).squeeze().numpy().astype(np.uint8)

        out = np.zeros((self.tiles_h * self.tile_hw, self.tiles_w * self.tile_hw), dtype=np.float32)
        offset = 0
        idx = 0
        for tr in range(self.tiles_h):
            for tc in range(self.tiles_w):
                mant_bits = self._tile_mant_bits[idx]
                size = bfp_tile_packed_size(mant_bits)
                tile = unpack_bfp_tile(flat_np[offset : offset + size], mant_bits)
                out[tr * self.tile_hw : (tr + 1) * self.tile_hw, tc * self.tile_hw : (tc + 1) * self.tile_hw] = tile
                offset += size
                idx += 1

        return torch.from_numpy(out).reshape(self.shape)

    def get_assignment_numpy(self) -> np.ndarray:
        """Get assignment as (tiles_h, tiles_w) numpy array."""
        assign_tensor = self.assignment
        if ttnn.is_tensor_storage_on_device(assign_tensor):
            assign_tensor = ttnn.from_device(assign_tensor)
        return ttnn.to_torch(assign_tensor).squeeze().numpy().astype(np.int8).reshape(self.tiles_h, self.tiles_w)

    @property
    def data_bytes(self) -> int:
        """Total packed data size in bytes."""
        return sum(bfp_tile_packed_size(mb) for mb in self._tile_mant_bits)

    @property
    def num_tiles(self) -> int:
        return self.tiles_h * self.tiles_w

    @property
    def tile_counts(self) -> dict[str, int]:
        """Count of tiles per format."""
        assign = self.get_assignment_numpy().ravel()
        counts = {fmt: 0 for fmt in COMPRESSED_FORMATS}
        for idx in assign:
            counts[COMPRESSED_FORMATS[idx]] += 1
        return counts

    def _pack(self, tensor: torch.Tensor, assignment: np.ndarray) -> tuple[torch.Tensor, list[int]]:
        """Pack tiles into flat uint8 buffer."""
        data_np = tensor.detach().float().cpu().numpy()
        chunks = []
        tile_mant_bits = []
        for tr in range(self.tiles_h):
            for tc in range(self.tiles_w):
                tile = data_np[
                    tr * self.tile_hw : (tr + 1) * self.tile_hw,
                    tc * self.tile_hw : (tc + 1) * self.tile_hw,
                ]
                fmt_idx = assignment[tr, tc]
                mant_bits = _FMT_IDX_TO_MANT_BITS[int(fmt_idx)]
                tile_mant_bits.append(mant_bits)
                chunks.append(pack_bfp_tile(tile, mant_bits))
        flat_np = np.concatenate(chunks)
        return torch.from_numpy(flat_np.copy()), tile_mant_bits

    def __repr__(self) -> str:
        counts = self.tile_counts
        fmt_str = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
        return (
            f"CompressedTensor(shape={self.shape}, tiles={self.num_tiles}, " f"data_bytes={self.data_bytes}, {fmt_str})"
        )
