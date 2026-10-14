# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise tilize's block knobs beyond their Phase 0 trivial values.

Shrinks L1_BLOCK_BUDGET_BYTES / sets BLOCK_HEIGHT_TILES via monkeypatch so the
multi-column-block, ragged-tail and multi-row-block paths run on small shapes.
"""

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd


@pytest.mark.parametrize(
    "shape, budget_tiles, block_height",
    [
        ((1, 1, 4096, 160), 2, None),  # C=5, bw=2 -> blocks 2,2,1 (ragged tail)
        ((1, 1, 2048, 128), 2, None),  # C=4, bw=2 -> two even blocks
        ((1, 1, 4096, 96), None, 3),  # row blocks of 3 tile-rows (ragged last row block)
        ((2, 3, 640, 160), 3, 2),  # leading-dim fold + both knobs
    ],
)
def test_tilize_knobs(device, monkeypatch, shape, budget_tiles, block_height):
    if budget_tiles is not None:
        per_col = pd.IN_DEPTH_TILE_ROWS * 2048 + pd.OUT_DEPTH_TILE_ROWS * 2048
        monkeypatch.setattr(pd, "L1_BLOCK_BUDGET_BYTES", budget_tiles * per_col)
    monkeypatch.setattr(pd, "BLOCK_HEIGHT_TILES", block_height)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.to_torch(tilize(tt_in))
    assert torch.equal(out, x)


@pytest.mark.parametrize(
    "shape, tile_h, knobs",
    [
        # Split reorder: the writer RISC copies half of every tile-row's sticks.
        ((1, 1, 16384, 64), 32, {"WRITER_COPY_SHARE": 0.5}),
        ((1, 1, 16384, 32), 32, {"WRITER_COPY_SHARE": 0.75}),
        ((1, 1, 8192, 64), 16, {"WRITER_COPY_SHARE": 0.5}),
        # Staging ring: 3 slots of 2 tile-rows, short first chunk, with and without the split.
        ((1, 1, 32768, 64), 32, {"STAGING_SLOTS": 3, "STAGING_SLOT_BUDGET_BYTES": 8192, "FIRST_CHUNK_TILE_ROWS": 1}),
        (
            (1, 1, 32768, 64),
            32,
            {
                "STAGING_SLOTS": 4,
                "STAGING_SLOT_BUDGET_BYTES": 4096,
                "FIRST_CHUNK_TILE_ROWS": 1,
                "WRITER_COPY_SHARE": 0.5,
            },
        ),
    ],
)
def test_tilize_coalesced_reader_knobs(device, monkeypatch, shape, tile_h, knobs):
    """Coalesced (bank-grouped staging) reader at non-default ring / split-reorder settings."""
    for name, value in knobs.items():
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, value)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.to_torch(tilize(tt_in, tile=ttnn.Tile([tile_h, 32])))
    assert torch.equal(out, x)
