# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side index tables for the adaLN tile-row gather: the fused norm reads tile row `tile_map[r]` of a
small expanded table (one row per table row, then one per tile straddling a run boundary) for input tile row `r`;
`onehot(expanded_indices) @ table` builds that table, so every tile the norm reads holds the per-token bits."""

from __future__ import annotations

import torch

TILE = 32
DEFAULT_MAX_MIXED_TILES = 16


def tilerow_remap(
    adaln_indices: torch.Tensor,
    *,
    num_rows: int,
    sp_factor: int,
    max_mixed_tiles: int = DEFAULT_MAX_MIXED_TILES,
    tile: int = TILE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """int32 `(tile_map, expanded_indices)`, each laid out so an even split hands every SP device its own slice;
    `adaln_indices` is the [S_padded] table row of every packed row (padding included), `num_rows` the table's R."""
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
    if int(tile_map.max()) >= num_rows + max_mixed_tiles:
        raise ValueError(
            f"tile map entry {int(tile_map.max())} exceeds the expanded table's {num_rows + max_mixed_tiles} tile rows"
        )
    return tile_map.reshape(-1).to(torch.int32), expanded.reshape(-1).to(torch.int32)
