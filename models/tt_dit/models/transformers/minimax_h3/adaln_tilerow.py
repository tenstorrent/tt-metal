# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side index tables for `MINIMAX_H3_ADALN_GATHER=tilerow`.

The packed sequence is a few long runs of one adaLN table row each, so nearly every 32-row tile selects a single
row. Instead of materialising a per-token [S, H] modulation, the fused norm reads tile row `tile_map[r]` of a
small expanded table for tile row `r` of its input:

    expanded tile row j < R    : table row j, repeated over all 32 rows
    expanded tile row R + k    : the rows of the k-th tile that straddles a run boundary, one per token

`expanded_indices` names the table row behind every expanded row, so `onehot(expanded_indices) @ table` builds the
expanded table with the same one-hot matmul the per-token gather uses: every tile the norm reads holds the same bits.
"""

from __future__ import annotations

import torch

TILE = 32
# Fixed so a trace serves every request of a bucket; each run boundary costs at most one mixed tile.
DEFAULT_MAX_MIXED_TILES = 16


def tilerow_remap(
    adaln_indices: torch.Tensor,
    *,
    num_rows: int,
    sp_factor: int,
    max_mixed_tiles: int = DEFAULT_MAX_MIXED_TILES,
    tile: int = TILE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-tile-row map and the expanded-table row indices for every SP device.

    adaln_indices: [S_padded] table row of every packed row, padding included; device d owns the d-th of
        `sp_factor` equal contiguous slices.
    num_rows: rows R of the modulation table (num_timesteps * MODALITY_NUM).

    Returns int32 `(tile_map [sp_factor * S_local / tile], expanded_indices [sp_factor * (R + max_mixed_tiles) * tile])`,
    each laid out so an even split of the last dim hands every SP device its own slice. Unused mixed slots hold row 0.
    """
    idx = adaln_indices.reshape(-1).to(torch.int64)
    if idx.numel() == 0 or idx.numel() % (sp_factor * tile):
        raise ValueError(f"{idx.numel()} rows do not split into {sp_factor} devices of whole {tile}-row tiles")
    if idx.min() < 0 or idx.max() >= num_rows:
        raise ValueError(f"adaLN indices span [{idx.min()}, {idx.max()}], the table has {num_rows} rows")

    tiles = idx.reshape(sp_factor, -1, tile)
    mixed = (tiles != tiles[..., :1]).any(dim=-1)
    worst = int(mixed.sum(dim=-1).max())
    if worst > max_mixed_tiles:
        raise ValueError(
            f"{worst} tiles on one device straddle an adaLN run boundary, above the {max_mixed_tiles} slots; "
            f"raise MINIMAX_H3_ADALN_MIXED_TILES"
        )

    slot = num_rows + torch.cumsum(mixed.to(torch.int64), dim=-1) - 1
    tile_map = torch.where(mixed, slot, tiles[..., 0])

    expanded = torch.zeros(sp_factor, (num_rows + max_mixed_tiles) * tile, dtype=torch.int64)
    expanded[:, : num_rows * tile] = torch.arange(num_rows).repeat_interleave(tile)
    for d in range(sp_factor):
        rows = tiles[d][mixed[d]].reshape(-1)
        expanded[d, num_rows * tile : num_rows * tile + rows.numel()] = rows
    return tile_map.reshape(-1).to(torch.int32), expanded.reshape(-1).to(torch.int32)
