# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blocking-model knobs of g6_fill stay correct at every setting the perf lamps measure.

MAX_FILL_CORES (the grid-sync lamp) and BLOCK_W_TILES on two production shapes (BB-22): [512]
(16 tiles) and [207] (7 tiles, ragged core split for most caps, partial last tile).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_fill import g6_fill, g6_fill_plan
from ttnn.operations.g6_fill import g6_fill_program_descriptor as program


@pytest.mark.parametrize("block_w_tiles", [None, 1, 3], ids=lambda b: f"block{b}")
@pytest.mark.parametrize("max_fill_cores", [None, 1, 2, 4, 8], ids=lambda c: f"cores{c}")
@pytest.mark.parametrize("n", [512, 207], ids=lambda n: f"N{n}")
def test_knobs_exact(device, monkeypatch, n, max_fill_cores, block_w_tiles):
    monkeypatch.setattr(program, "MAX_FILL_CORES", max_fill_cores)
    monkeypatch.setattr(program, "BLOCK_W_TILES", block_w_tiles)
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(n) + 10.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.from_torch(torch.randn(n) + 10.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    plan = g6_fill_plan(x)
    tensor_w_tiles = -(-n // 32)
    grid = device.compute_with_storage_grid_size()
    cap = grid.x * grid.y if max_fill_cores is None else max_fill_cores
    assert plan["num_cores"] == min(tensor_w_tiles, cap)
    assert sum(plan["tiles_per_core"].values()) == tensor_w_tiles

    for value in (1.0, -0.0, 0.1):
        g6_fill(x, value, output_tensor=out)
        expected = torch.tensor(value, dtype=torch.float64).to(torch.float32).view(torch.int32)
        got = ttnn.to_torch(out).view(torch.int32)
        assert torch.equal(got, expected.expand(n)), f"fill_value={value!r}"
