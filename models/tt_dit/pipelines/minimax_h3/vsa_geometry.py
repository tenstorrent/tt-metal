# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""VSA tile geometry for MiniMax-H3 (R1 of VSA_SCOPE.md).

Host-side and pure torch. Reorders the packed ``[text | cond | audio | video]``
sequence into 64-token VSA tiles:

- video rows are partitioned cube-major into ``(4, 4, 4)`` tiles of the
  ``(t, h, w)`` token grid, ragged edges allowed (partial tiles at each
  dimension's tail);
- every prefix segment (text, condition keyframes, audio) is chopped into
  64-token chunks that never cross a segment boundary;
- ragged tiles are zero-padded to 64 slots, and whole zero-valid pad tiles are
  appended so the tile count divides the SP factor -- per-device sequence
  length is a multiple of 64 and no tile straddles an SP shard boundary.

The video partition and valid-count math is a port of FastVideo's geometry
builders (``get_tile_partition_indices`` / ``construct_variable_block_sizes``,
vendored untouched under ``models/tt_dit/models/transformers/minimax_h3/
vsa_reference/``) and is cross-checked against them by
``test_vsa_geometry_minimax_h3.py``.

Unlike FastVideo, which scatters into a padded tile buffer inside each
attention call, the tiled order here IS the device-resident sequence: the
reorder happens once on host at model input, RoPE tables and AdaLN row indices
follow the same permutation, and the output is unpacked once at the end. Pad
slots therefore carry garbage activations through the model; correctness never
depends on their values -- the fine stage masks pad columns via per-block
valid counts, the coarse stage pools through a host-built averaging matrix
whose pad columns are zero, and pad-row outputs are dropped at unpacking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

VSA_TILE_SHAPE = (4, 4, 4)
VSA_TILE_TOKENS = 64  # math.prod(VSA_TILE_SHAPE); also the fine-stage block size

VSA_PLACEMENTS = ("identity", "striped")


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
    """One packed sequence's tile geometry, in final device (placement) order.

    All per-tile tensors are indexed by placement slot. ``tile_ids`` maps a
    placement slot back to the tile's canonical FastVideo-order id (prefix
    chunks first, then cube-major video tiles; -1 for appended pad tiles);
    the R3/R4 index tensors use placement-order ids, matching the gathered
    K/V layout on device.
    """

    sp_factor: int
    placement: str
    seq_len: int  # packed rows before tiling
    n_prefix_tiles: int  # counts by kind, placement-independent
    n_video_tiles: int
    n_pad_tiles: int

    valid_counts: torch.Tensor  # [n_tiles] long; 0 for pad tiles
    tile_ids: torch.Tensor  # [n_tiles] long; canonical id, -1 for pad tiles
    is_3d: torch.Tensor  # [n_tiles] bool; video tiles
    is_exempt: torch.Tensor  # [n_tiles] bool; always-attended keys, dense queries
    is_candidate: torch.Tensor  # [n_tiles] bool; compete in top-k (excludes pad tiles)

    gather_index: torch.Tensor  # [n_tiles * 64] long; source packed row per slot, -1 for pad slots
    untile_index: torch.Tensor  # [seq_len] long; padded slot per packed row (inverse of the above)
    row_source: torch.Tensor  # [n_tiles * 64] long; gather_index with pad slots clamped to a valid
    # row of the same tile (row 0 for pure pad tiles) -- for permuting
    # per-row metadata (RoPE tables, AdaLN indices) where pad values are
    # dont-cares but must keep runs compressed

    @property
    def n_tiles(self) -> int:
        return int(self.valid_counts.numel())

    @property
    def padded_len(self) -> int:
        return self.n_tiles * VSA_TILE_TOKENS

    @property
    def tiles_per_shard(self) -> int:
        return self.n_tiles // self.sp_factor

    def pack_rows(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Reorder packed rows into tile order along ``dim``, zero at pad slots."""
        x = x.movedim(dim, 0)
        out = torch.zeros((self.padded_len, *x.shape[1:]), dtype=x.dtype, device=x.device)
        out[self.gather_index >= 0] = x[self.gather_index[self.gather_index >= 0]]
        return out.movedim(0, dim)

    def unpack_rows(self, y: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Inverse of :meth:`pack_rows`: recover original packed order, dropping pads."""
        y = y.movedim(dim, 0)
        return y[self.untile_index].movedim(0, dim)

    def permute_metadata(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Reorder per-row metadata into tile order; pad slots replicate a valid row of their tile."""
        return x.movedim(dim, 0)[self.row_source].movedim(0, dim)

    def averaging_matrix(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """[n_tiles, padded_len] block-diagonal masked-mean matrix (R3a).

        Row ``i`` holds ``1 / valid_counts[i]`` on tile ``i``'s valid slots and
        zero elsewhere (entire row zero for pad tiles), so ``A @ X`` is the
        masked mean of each tile's rows regardless of pad-slot values.
        """
        slots = torch.arange(self.padded_len)
        tile = slots // VSA_TILE_TOKENS
        valid = (slots % VSA_TILE_TOKENS) < self.valid_counts[tile]
        weights = torch.zeros(self.n_tiles, self.padded_len, dtype=torch.float64)
        weights[tile[valid], slots[valid]] = 1.0 / self.valid_counts[tile[valid]].to(torch.float64)
        return weights.to(dtype)


def _striped_order(is_exempt: torch.Tensor, sp_factor: int, capacity: int) -> torch.Tensor:
    """Canonical tile ids in striped placement order.

    Exempt tiles are dealt round-robin across shards (in canonical order) so no
    shard concentrates the dense-query stragglers; the remaining tiles
    (candidates, then pad tiles, in canonical order) fill each shard's leftover
    capacity in shard order. Within a shard, exempt tiles come first.
    """
    n_tiles = int(is_exempt.numel())
    exempt_ids = torch.nonzero(is_exempt, as_tuple=False).reshape(-1)
    other_ids = torch.nonzero(~is_exempt, as_tuple=False).reshape(-1)

    shards: list[list[int]] = [[] for _ in range(sp_factor)]
    for i, tile in enumerate(exempt_ids.tolist()):
        shards[i % sp_factor].append(tile)
    if any(len(s) > capacity for s in shards):
        raise ValueError(f"{exempt_ids.numel()} exempt tiles cannot stripe over {sp_factor} shards of {capacity}")
    queue = other_ids.tolist()
    for shard in shards:
        take = capacity - len(shard)
        shard.extend(queue[:take])
        queue = queue[take:]
    assert not queue
    order = torch.tensor([tile for shard in shards for tile in shard], dtype=torch.long)
    assert order.numel() == n_tiles
    return order


def build_vsa_geometry(
    prefix_segments: tuple[int, ...],
    video_grid: tuple[int, int, int],
    *,
    sp_factor: int,
    placement: str = "identity",
) -> MiniMaxH3VSAGeometry:
    """Build the tile geometry for one packed ``[prefix... | video]`` sequence.

    ``prefix_segments`` are the 1D segment lengths preceding the generated
    video, in packed order -- ``(n_text, n_cond, n_audio)`` for the standard
    t2va/fl2va layout, matching FastVideo's ``_h3_vsa_prefix_segments``.
    ``video_grid`` is the ``(t, h, w)`` token grid (latents over patch size).

    v0 policy: every prefix tile is 1D and exempt; every video tile is 3D and
    a top-k candidate; pad tiles are neither exempt nor candidates.
    """
    if placement not in VSA_PLACEMENTS:
        raise ValueError(f"placement must be one of {VSA_PLACEMENTS}, got {placement!r}")
    if sp_factor < 1:
        raise ValueError(f"sp_factor must be positive, got {sp_factor}")
    if any(d < 1 for d in video_grid):
        raise ValueError(f"video grid must be positive, got {video_grid}")

    prefix_len = sum(prefix_segments)
    n_video_rows = math.prod(video_grid)
    seq_len = prefix_len + n_video_rows

    prefix_sizes = chop_prefix_segments(prefix_segments)
    n_prefix_tiles = len(prefix_sizes)
    video_counts = video_tile_valid_counts(video_grid)
    n_video_tiles = int(video_counts.numel())

    n_real_tiles = n_prefix_tiles + n_video_tiles
    n_tiles = math.ceil(n_real_tiles / sp_factor) * sp_factor
    n_pad_tiles = n_tiles - n_real_tiles

    # Canonical (FastVideo-order) per-tile metadata, pad tiles appended.
    valid_counts = torch.cat(
        [torch.tensor(prefix_sizes, dtype=torch.long), video_counts, torch.zeros(n_pad_tiles, dtype=torch.long)]
    )
    is_3d = torch.cat(
        [
            torch.zeros(n_prefix_tiles, dtype=torch.bool),
            torch.ones(n_video_tiles, dtype=torch.bool),
            torch.zeros(n_pad_tiles, dtype=torch.bool),
        ]
    )
    is_exempt = torch.cat(
        [torch.ones(n_prefix_tiles, dtype=torch.bool), torch.zeros(n_video_tiles + n_pad_tiles, dtype=torch.bool)]
    )
    is_candidate = ~is_exempt & (valid_counts > 0)

    # Per-tile source rows in canonical order: prefix chunks are contiguous
    # runs of [0, prefix_len); video tiles follow the cube-major partition.
    video_rows = video_tile_partition_indices(video_grid) + prefix_len
    canonical_rows: list[torch.Tensor] = []
    start = 0
    for size in prefix_sizes:
        canonical_rows.append(torch.arange(start, start + size, dtype=torch.long))
        start += size
    start = 0
    for count in video_counts.tolist():
        canonical_rows.append(video_rows[start : start + count])
        start += count
    canonical_rows.extend(torch.empty(0, dtype=torch.long) for _ in range(n_pad_tiles))

    if placement == "identity":
        order = torch.arange(n_tiles, dtype=torch.long)
    else:
        order = _striped_order(is_exempt, sp_factor, n_tiles // sp_factor)

    valid_counts = valid_counts[order]
    is_3d = is_3d[order]
    is_exempt = is_exempt[order]
    is_candidate = is_candidate[order]
    tile_ids = torch.where(order < n_real_tiles, order, torch.full_like(order, -1))

    gather_index = torch.full((n_tiles * VSA_TILE_TOKENS,), -1, dtype=torch.long)
    row_source = torch.zeros(n_tiles * VSA_TILE_TOKENS, dtype=torch.long)
    for slot, tile in enumerate(order.tolist()):
        rows = canonical_rows[tile]
        base = slot * VSA_TILE_TOKENS
        gather_index[base : base + rows.numel()] = rows
        if rows.numel():
            row_source[base : base + VSA_TILE_TOKENS] = rows[0]
            row_source[base : base + rows.numel()] = rows

    untile_index = torch.empty(seq_len, dtype=torch.long)
    valid_slots = torch.nonzero(gather_index >= 0, as_tuple=False).reshape(-1)
    untile_index[gather_index[valid_slots]] = valid_slots

    return MiniMaxH3VSAGeometry(
        sp_factor=sp_factor,
        placement=placement,
        seq_len=seq_len,
        n_prefix_tiles=n_prefix_tiles,
        n_video_tiles=n_video_tiles,
        n_pad_tiles=n_pad_tiles,
        valid_counts=valid_counts,
        tile_ids=tile_ids,
        is_3d=is_3d,
        is_exempt=is_exempt,
        is_candidate=is_candidate,
        gather_index=gather_index,
        untile_index=untile_index,
        row_source=row_source,
    )
