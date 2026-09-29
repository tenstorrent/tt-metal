# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gathering rows of a tiled table must match host row indexing bit for bit."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()


@pytest.mark.parametrize(
    "indices",
    [
        # Face and tile boundaries: rows 15/16 cross a face, 31/32 cross a tile row.
        [15, 16, 17],
        [29, 30, 31],
        [31, 32, 33],
        # Out of order, repeated and up to the 16-row limit.
        [639, 0, 320, 320, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ],
    ids=["face-boundary", "tile-tail", "tile-boundary", "sixteen-rows"],
)
# The table is 12440 columns wide (padded to 12448); 12416 ends at its last full tile.
@pytest.mark.parametrize("width", [9216, 12416], ids=["leading-columns", "last-full-tile"])
def test_select_tile_rows_matches_host_indexing(device, indices, width):
    rows, columns = 640, 12440
    table = torch.randn(1, rows, columns, generator=torch.Generator().manual_seed(7)).bfloat16()
    table_tt = ttnn.from_torch(table, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    indices_tt = ttnn.from_torch(
        torch.tensor(indices, dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    selected = ttnn.experimental.kda.select_tile_rows(table_tt, indices_tt, width=width)

    assert selected.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tuple(selected.shape) == (1, len(indices), width)
    assert torch.equal(ttnn.to_torch(selected), table[:, indices, :width])
