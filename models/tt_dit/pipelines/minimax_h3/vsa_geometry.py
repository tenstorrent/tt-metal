# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""VSA tile geometry for MiniMax-H3.

Host-side and pure torch. Lays the packed ``[text | cond | audio | video]``
sequence out as 64-token VSA tiles inside a ladder rung:

- video rows are partitioned cube-major into ``(4, 4, 4)`` tiles of the
  ``(t, h, w)`` token grid, ragged edges allowed (partial tiles at each
  dimension's tail);
- every prefix segment (text, condition keyframes, audio) is chopped into
  64-token chunks that never cross a segment boundary;
- each SP shard holds ``rung / sp`` rows as 64-row blocks; the real tiles fill
  the whole blocks of shard 0, then shard 1, and so on. Every other block is a
  pad block with count 0: unused blocks, and the 32-row block that ends each
  shard when ``rung / sp`` is an odd multiple of 32.

The rung comes from the bucket ladder (``vsa_min_padded_len`` is what it must
hold), so the tile layout never changes the rung a request runs at.

The video partition and valid-count math is a port of FastVideo's geometry
builders (``get_tile_partition_indices`` / ``construct_variable_block_sizes``,
vendored under ``models/tt_dit/tests/models/minimax_h3/vsa_reference/``) and
is cross-checked against them by ``test_vsa_geometry_minimax_h3.py``.

Unlike FastVideo, which scatters into a padded tile buffer inside each
attention call, this layout IS the device-resident sequence: the reorder
happens once on host at model input, RoPE tables and AdaLN row indices follow
the same permutation, and the output is unpacked once at the end. Pad rows
therefore carry garbage activations through the model; correctness never
depends on their values -- the fine stage masks them via per-block valid
counts, the coarse stage pools through a host-built averaging matrix whose pad
columns are zero, and pad-row outputs are dropped at unpacking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

VSA_TILE_SHAPE = (4, 4, 4)
# math.prod(VSA_TILE_SHAPE); also the fine-stage block size.
VSA_TILE_TOKENS = 64


def chop_prefix_segments(prefix_segments: tuple[int, ...]) -> list[int]:
    """Chunk sizes for the 1D prefix, segment-pure (a chunk never crosses a boundary).

    Mirrors the prefix loop of FastVideo's ``_h3_tile_geometry``: each segment
    contributes ``segment // 64`` full chunks plus one ragged tail chunk.
    Zero-length segments are skipped, as upstream's builder does.
    """
    sizes: list[int] = []
    for segment in prefix_segments:
        if segment < 0:
            raise ValueError(f"prefix segments must be non-negative, got {prefix_segments}")
        full, rem = divmod(segment, VSA_TILE_TOKENS)
        sizes.extend([VSA_TILE_TOKENS] * full)
        if rem:
            sizes.append(rem)
    return sizes


def video_tile_partition_indices(video_grid: tuple[int, int, int]) -> torch.Tensor:
    """Video-row order after cube-major (4, 4, 4) tiling; ragged tails allowed.

    Port of FastVideo's ``get_tile_partition_indices``: rows of the ``(t, h, w)``
    token grid, concatenated tile by tile in (t-tile, h-tile, w-tile) order,
    row-major within each tile. Values index into the frame-major packed video
    rows (row of ``(t, h, w)`` = ``t*H*W + h*W + w``).
    """
    t, h, w = video_grid
    ts, hs, ws = VSA_TILE_SHAPE
    indices = torch.arange(t * h * w, dtype=torch.long).reshape(t, h, w)
    tiles = []
    for tt in range(math.ceil(t / ts)):
        for hh in range(math.ceil(h / hs)):
            for ww in range(math.ceil(w / ws)):
                tiles.append(indices[tt * ts : tt * ts + ts, hh * hs : hh * hs + hs, ww * ws : ww * ws + ws].flatten())
    return torch.cat(tiles)


def video_tile_valid_counts(video_grid: tuple[int, int, int]) -> torch.Tensor:
    """Valid (non-pad) tokens per cube-major video tile.

    Port of FastVideo's ``construct_variable_block_sizes``: the per-dimension
    tail sizes broadcast-multiplied, flattened (t-tile, h-tile, w-tile)-major.
    """

    def _sizes(dim_len: int, tile: int) -> torch.Tensor:
        n = math.ceil(dim_len / tile)
        sizes = torch.full((n,), tile, dtype=torch.long)
        remainder = dim_len - (n - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(video_grid[0], VSA_TILE_SHAPE[0])
    h_sizes = _sizes(video_grid[1], VSA_TILE_SHAPE[1])
    w_sizes = _sizes(video_grid[2], VSA_TILE_SHAPE[2])
    return (t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]).reshape(-1)


@dataclass
class MiniMaxH3VSAGeometry:
    """One packed sequence's tiles laid out in a ``padded_len``-row rung, in device row order.

    Per-block tensors are indexed by global block ``s * blocks_per_shard + b``, block ``b`` of shard ``s``, which
    starts at row ``s * rows_per_shard + 64 * b``.
    """

    sp_factor: int
    padded_len: int
    video_grid: tuple[int, int, int]
    # Packed rows before the layout.
    seq_len: int
    n_prefix_tiles: int
    n_video_tiles: int

    # Real tokens per block; 0 for a pad block.
    valid_counts: torch.Tensor
    # Prefix tiles: always-attended keys, dense queries.
    is_exempt: torch.Tensor
    # Video tiles: compete in top-k.
    is_candidate: torch.Tensor

    # Packed source row of each device row, -1 for a pad row.
    gather_index: torch.Tensor
    # Device row of each packed row (inverse of the above).
    untile_index: torch.Tensor
    # gather_index with pad rows pointing at a real row of their tile (row 0 in a pad block), for permuting
    # per-row metadata (RoPE tables, AdaLN indices) whose pad values are don't-cares but must keep runs compressed.
    row_source: torch.Tensor

    @property
    def rows_per_shard(self) -> int:
        return self.padded_len // self.sp_factor

    @property
    def blocks_per_shard(self) -> int:
        return -(-self.rows_per_shard // VSA_TILE_TOKENS)

    @property
    def n_blocks(self) -> int:
        return self.sp_factor * self.blocks_per_shard

    @property
    def n_real_tiles(self) -> int:
        return self.n_prefix_tiles + self.n_video_tiles

    @property
    def block_starts(self) -> torch.Tensor:
        block = torch.arange(self.n_blocks)
        shard, slot = block // self.blocks_per_shard, block % self.blocks_per_shard
        return shard * self.rows_per_shard + slot * VSA_TILE_TOKENS

    @property
    def block_of_row(self) -> torch.Tensor:
        row = torch.arange(self.padded_len)
        return (row // self.rows_per_shard) * self.blocks_per_shard + (row % self.rows_per_shard) // VSA_TILE_TOKENS

    @property
    def logical_n(self) -> int:
        """Row just past the last real tile's block; the ring op masks the tail beyond it."""
        shard, slot = divmod(self.n_real_tiles - 1, self.rows_per_shard // VSA_TILE_TOKENS)
        return shard * self.rows_per_shard + (slot + 1) * VSA_TILE_TOKENS

    def pack_rows(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Reorder packed rows into device row order along ``dim``, zero at pad rows."""
        x = x.movedim(dim, 0)
        out = torch.zeros((self.padded_len, *x.shape[1:]), dtype=x.dtype, device=x.device)
        out[self.gather_index >= 0] = x[self.gather_index[self.gather_index >= 0]]
        return out.movedim(0, dim)

    def unpack_rows(self, y: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Inverse of :meth:`pack_rows`: recover original packed order, dropping pads."""
        y = y.movedim(dim, 0)
        return y[self.untile_index].movedim(0, dim)

    def permute_metadata(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Reorder per-row metadata into device row order; pad rows replicate a valid row of their tile."""
        return x.movedim(dim, 0)[self.row_source].movedim(0, dim)

    def averaging_matrix(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """[n_blocks, padded_len] masked-mean matrix.

        Row ``i`` holds ``1 / valid_counts[i]`` on block ``i``'s real rows and
        zero elsewhere (entire row zero for pad blocks), so ``A @ X`` is the
        masked mean of each tile's rows regardless of pad-row values.
        """
        block = self.block_of_row
        valid = torch.arange(self.padded_len) - self.block_starts[block] < self.valid_counts[block]
        rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        weights = torch.zeros(self.n_blocks, self.padded_len, dtype=torch.float64)
        weights[block[rows], rows] = 1.0 / self.valid_counts[block[rows]].to(torch.float64)
        return weights.to(dtype)


def vsa_min_padded_len(prefix_segments: tuple[int, ...], video_grid: tuple[int, int, int], sp_factor: int) -> int:
    """The shortest padded length whose whole 64-row blocks hold every tile: the length to select a rung with."""
    n_tiles = len(chop_prefix_segments(prefix_segments)) + int(video_tile_valid_counts(video_grid).numel())
    return VSA_TILE_TOKENS * sp_factor * math.ceil(n_tiles / sp_factor)


def build_vsa_geometry(
    prefix_segments: tuple[int, ...],
    video_grid: tuple[int, int, int],
    *,
    sp_factor: int,
    padded_len: int,
) -> MiniMaxH3VSAGeometry:
    """Lay one packed ``[prefix... | video]`` sequence out in a ``padded_len``-row rung.

    ``prefix_segments`` are the 1D segment lengths preceding the generated
    video, in packed order -- ``(n_text, n_cond, n_audio)`` for the standard
    t2va/fl2va layout, matching FastVideo's ``_h3_vsa_prefix_segments``.
    ``video_grid`` is the ``(t, h, w)`` token grid (latents over patch size).

    v0 policy: every prefix tile is 1D and exempt; every video tile is 3D and
    a top-k candidate; pad blocks are neither.
    """
    if sp_factor < 1:
        raise ValueError(f"sp_factor must be positive, got {sp_factor}")
    if any(d < 1 for d in video_grid):
        raise ValueError(f"video grid must be positive, got {video_grid}")
    if padded_len % (sp_factor * 32):
        raise ValueError(f"padded_len {padded_len} is not a multiple of sp_factor * 32 = {sp_factor * 32}")
    min_len = vsa_min_padded_len(prefix_segments, video_grid, sp_factor)
    if padded_len < min_len:
        raise ValueError(f"padded_len {padded_len} holds fewer whole blocks than the tiles need ({min_len} rows)")

    prefix_len = sum(prefix_segments)
    seq_len = prefix_len + math.prod(video_grid)
    prefix_sizes = chop_prefix_segments(prefix_segments)
    n_prefix_tiles = len(prefix_sizes)
    video_counts = video_tile_valid_counts(video_grid)
    n_video_tiles = int(video_counts.numel())
    n_real_tiles = n_prefix_tiles + n_video_tiles

    rows_per_shard = padded_len // sp_factor
    slots_per_shard = rows_per_shard // VSA_TILE_TOKENS
    blocks_per_shard = -(-rows_per_shard // VSA_TILE_TOKENS)
    n_blocks = sp_factor * blocks_per_shard
    tile = torch.arange(n_real_tiles)
    shard, slot = tile // slots_per_shard, tile % slots_per_shard
    tile_block = shard * blocks_per_shard + slot
    tile_start = shard * rows_per_shard + slot * VSA_TILE_TOKENS

    valid_counts = torch.zeros(n_blocks, dtype=torch.long)
    valid_counts[tile_block] = torch.cat([torch.tensor(prefix_sizes, dtype=torch.long), video_counts])
    is_exempt = torch.zeros(n_blocks, dtype=torch.bool)
    is_exempt[tile_block[:n_prefix_tiles]] = True
    is_candidate = torch.zeros(n_blocks, dtype=torch.bool)
    is_candidate[tile_block[n_prefix_tiles:]] = True

    # Source rows of each tile in FastVideo order: prefix chunks are contiguous runs of [0, prefix_len); video
    # tiles follow the cube-major partition.
    video_rows = video_tile_partition_indices(video_grid) + prefix_len
    tile_rows = list(torch.arange(prefix_len, dtype=torch.long).split(prefix_sizes))
    tile_rows += list(video_rows.split(video_counts.tolist()))

    gather_index = torch.full((padded_len,), -1, dtype=torch.long)
    row_source = torch.zeros(padded_len, dtype=torch.long)
    for start, rows in zip(tile_start.tolist(), tile_rows):
        gather_index[start : start + rows.numel()] = rows
        row_source[start : start + VSA_TILE_TOKENS] = rows[0]
        row_source[start : start + rows.numel()] = rows

    untile_index = torch.empty(seq_len, dtype=torch.long)
    valid_rows = torch.nonzero(gather_index >= 0, as_tuple=False).reshape(-1)
    untile_index[gather_index[valid_rows]] = valid_rows

    return MiniMaxH3VSAGeometry(
        sp_factor=sp_factor,
        padded_len=padded_len,
        video_grid=tuple(video_grid),
        seq_len=seq_len,
        n_prefix_tiles=n_prefix_tiles,
        n_video_tiles=n_video_tiles,
        valid_counts=valid_counts,
        is_exempt=is_exempt,
        is_candidate=is_candidate,
        gather_index=gather_index,
        untile_index=untile_index,
        row_source=row_source,
    )
