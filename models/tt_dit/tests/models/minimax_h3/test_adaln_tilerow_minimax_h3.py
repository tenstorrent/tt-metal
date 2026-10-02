# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only exactness check for MINIMAX_H3_ADALN_GATHER=tilerow (no device): reading tile row tile_map[r] of the
expanded table, page by page as the fused norm's reader does, reproduces the per-token gather bit for bit."""

import pytest
import torch

from ....models.transformers.minimax_h3.adaln_tilerow import DEFAULT_MAX_MIXED_TILES, TILE, tilerow_remap
from ....pipelines.minimax_h3 import packing as p

HIDDEN = 1344


def _reader_gather(table: torch.Tensor, tile_map: torch.Tensor, expanded: torch.Tensor, sp_factor: int) -> torch.Tensor:
    """Per device: the expanded table is `table[expanded]` (what onehot(expanded) @ table writes), split into tile
    pages row-major; input tile (r, c) reads page tile_map[r] * num_tile_cols + c."""
    cols = table.shape[1] // TILE
    out = []
    for m, e in zip(tile_map.chunk(sp_factor), expanded.chunk(sp_factor)):
        rows = table[e.long()]
        pages = rows.reshape(-1, TILE, cols, TILE).permute(0, 2, 1, 3).reshape(-1, TILE, TILE)
        page_ids = m.long()[:, None] * cols + torch.arange(cols)[None, :]
        tiles = pages[page_ids]
        out.append(tiles.permute(0, 2, 1, 3).reshape(-1, table.shape[1]))
    return torch.cat(out)


def _check(indices: torch.Tensor, num_rows: int, sp_factor: int) -> int:
    table = torch.randn(num_rows, HIDDEN).to(torch.bfloat16)
    tile_map, expanded = tilerow_remap(indices, num_rows=num_rows, sp_factor=sp_factor)
    assert tile_map.shape[0] == indices.shape[0] // TILE
    assert expanded.shape[0] == sp_factor * (num_rows + DEFAULT_MAX_MIXED_TILES) * TILE
    got = _reader_gather(table, tile_map, expanded, sp_factor)
    assert torch.equal(got.view(torch.int16), table[indices.long()].view(torch.int16))
    return int((tile_map >= num_rows).sum())


def _runs(lengths_and_rows, pad_to):
    idx = torch.cat([torch.full((n,), r, dtype=torch.int64) for n, r in lengths_and_rows])
    return torch.cat([idx, torch.zeros(pad_to - idx.shape[0], dtype=torch.int64)])


@pytest.mark.parametrize("sp_factor", [1, 2, 4])
def test_synthetic_runs_with_mid_tile_boundaries(sp_factor):
    runs = [(45, 1), (19, 0), (64, 1), (5, 3), (300, 2), (700, 5), (31, 4), (33, 0)]
    total = sum(n for n, _ in runs)
    align = sp_factor * TILE
    indices = _runs(runs, ((total + align - 1) // align + 1) * align)
    mixed = _check(indices, num_rows=6, sp_factor=sp_factor)
    assert mixed > 0


def test_random_runs_match_per_token_gather():
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        runs = [
            (int(torch.randint(1, 400, (1,), generator=gen)), int(torch.randint(0, 12, (1,), generator=gen)))
            for _ in range(6)
        ]
        total = sum(n for n, _ in runs)
        indices = _runs(runs, ((total + 8 * TILE - 1) // (8 * TILE)) * 8 * TILE)
        _check(indices, num_rows=12, sp_factor=8)


@pytest.mark.parametrize("sp_factor", [8, 32])
def test_packed_fl2va_layout(sp_factor):
    tags = torch.ones(997, dtype=torch.long)
    tags[20:70] = p.MINIMAX_H3_VIDEO_TAG
    layout = p.build_packed_sequence(
        tags,
        p.video_latent_num_frames(124),
        544 // 16,
        960 // 16,
        p.audio_latent_num_frames(124),
        (1, 2, 2),
        ("first", "last"),
    )
    row_slot, roles = p.build_slot_routing(layout)
    rows = p.adaln_indices(layout.token_tags, row_slot)
    align = sp_factor * TILE
    padded = torch.cat([rows, torch.zeros((-rows.shape[0]) % align, dtype=rows.dtype)])
    mixed = _check(padded, num_rows=len(roles) * p.MINIMAX_H3_MODALITY_NUM, sp_factor=sp_factor)
    assert 0 < mixed <= 8


def test_too_many_mixed_tiles_raises():
    indices = torch.arange(4 * TILE) % 2
    with pytest.raises(  # allow-pytest.raises: the message is the knob's contract
        ValueError, match="MINIMAX_H3_ADALN_MIXED_TILES"
    ):
        tilerow_remap(indices, num_rows=2, sp_factor=1, max_mixed_tiles=3)
