# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Knob matrix for tilize: every parked lever stays correct at non-default values.

The default path is covered by test_tilize.py; this turns the host knobs in
tilize_program_descriptor.py (READ_AHEAD, SPLIT_READER_MAX_SEGMENT_BYTES,
CB_BUDGET_BYTES -> block_width) so the ragged column block, multi-column-block
walks, the transaction-id read-ahead and the split reader (odd/even walk
positions over two input CBs) are all exercised. Output must be bit-identical.
"""
import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

SHAPES = [
    (1, 1, 32, 32),  # 1 position: split reader must fall back
    (1, 1, 64, 160),  # R=2, C=5: ragged column block under the tiny budget
    (1, 1, 96, 160),  # R=3, C=5: odd tile-row count per core
    (2, 3, 64, 96),  # leading-dim fold
    (1, 1, 1600, 96),  # R=50 uneven row split
    (1, 1, 16384, 64),  # 8 tile-rows per core
]

CONFIGS = {
    "read_ahead2": dict(READ_AHEAD=2),
    "split": dict(SPLIT_READER_MAX_SEGMENT_BYTES=1 << 20),
    "split_read_ahead2": dict(SPLIT_READER_MAX_SEGMENT_BYTES=1 << 20, READ_AHEAD=2),
    "tiny_budget": dict(CB_BUDGET_BYTES={False: 3 * 8192, True: 3 * 8192}),
    "tiny_budget_split_ra2": dict(
        CB_BUDGET_BYTES={False: 3 * 12288, True: 3 * 12288},
        SPLIT_READER_MAX_SEGMENT_BYTES=1 << 20,
        READ_AHEAD=2,
    ),
    "depth3_ra3": dict(DEPTH_IN=3, READ_AHEAD=3),
}


@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_tilize_knob(device, monkeypatch, shape, config):
    for name, value in CONFIGS[config].items():
        monkeypatch.setattr(pd, name, value)
    torch.manual_seed(7)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = tilize(t)
    assert out.layout == ttnn.TILE_LAYOUT
    y = ttnn.to_torch(out)
    mismatches = int((y != x).sum())
    assert mismatches == 0, f"{mismatches} of {x.numel()} elements differ"
