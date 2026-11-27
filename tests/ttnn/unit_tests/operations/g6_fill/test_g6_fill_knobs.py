# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blocking-model knobs of g6_fill stay live: non-default core counts and block sizes stay exact.

Production shapes only (BB-22): [512] (16 tiles, the PERF-FOCUS case) and [207] (7 tiles,
partial last tile). Also checks the plan report against the dispatched split (BB-08, BB-17).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_fill import g6_fill, g6_fill_plan
from ttnn.operations.g6_fill import g6_fill_program_descriptor as pd


def _fill_and_check(device, n, value):
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(n),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.randn(n) + 10.0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    result = g6_fill(x, value, output_tensor=out)
    got = ttnn.to_torch(result).view(torch.int32)
    want = torch.full((n,), value, dtype=torch.float32).view(torch.int32)
    assert torch.equal(got, want), f"N={n}: {(got != want).sum().item()} elements differ"


@pytest.mark.parametrize("n", [512, 207], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("max_fill_cores", [1, 2, 4, 8])
@pytest.mark.parametrize("block_w_tiles", [None, 1, 3])
def test_knobs_exact(device, monkeypatch, n, max_fill_cores, block_w_tiles):
    monkeypatch.setattr(pd, "MAX_FILL_CORES", max_fill_cores)
    monkeypatch.setattr(pd, "BLOCK_W_TILES", block_w_tiles)
    _fill_and_check(device, n, 1 / 3)


def test_plan_matches_design(device):
    x = ttnn.from_torch(
        torch.zeros(512),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    plan = g6_fill_plan(x)
    grid = device.compute_with_storage_grid_size()
    assert plan["regime"] == "R1_tile_row0_segments"
    assert plan["num_cores"] == min(16, grid.x * grid.y)
    assert sum(plan["tiles_per_core"].values()) == 16
    assert plan["segments_per_tile"] == 2 and plan["segment_bytes"] == 64
    if grid.x == 11:
        assert plan["core_ranges"] == [((0, 0), (10, 0)), ((0, 1), (4, 1))]
