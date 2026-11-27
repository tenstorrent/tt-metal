# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blocking-knob settings of g6_fill on the production shapes: each setting must stay bit-exact.

Under --profile, one call per setting gives the device kernel time of that plan
(grid-sync lamp: active-core cap; aspect lamp: rectangle-width cap; block size).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_fill import g6_fill, g6_fill_plan
from ttnn.operations.g6_fill import g6_fill_program_descriptor as pd

# (max_active_cores, max_rect_w, block_w_tiles)
KNOBS = [
    pytest.param((None, None, None), id="default"),
    pytest.param((16, 4, None), id="c16_w4"),
    pytest.param((8, None, None), id="c8"),
    pytest.param((8, 4, None), id="c8_w4"),
    pytest.param((4, None, None), id="c4"),
    pytest.param((4, 2, None), id="c4_w2"),
    pytest.param((2, None, None), id="c2"),
    pytest.param((1, None, None), id="c1"),
    pytest.param((1, None, 1), id="c1_b1"),
    pytest.param((4, None, 1), id="c4_b1"),
]


@pytest.mark.parametrize("knobs", KNOBS)
@pytest.mark.parametrize("shape", [(512,), (185,)], ids=lambda s: str(s[0]))
def test_fill_knobs(device, monkeypatch, shape, knobs):
    max_active_cores, max_rect_w, block_w_tiles = knobs
    monkeypatch.setattr(pd, "MAX_ACTIVE_CORES", max_active_cores)
    monkeypatch.setattr(pd, "MAX_RECT_W", max_rect_w)
    monkeypatch.setattr(pd, "BLOCK_W_TILES", block_w_tiles)

    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(shape),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.randn(shape) + 10.0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    value = 1 / 3
    g6_fill(x, value, output_tensor=out)

    plan = g6_fill_plan(x)
    expected = torch.full(shape, value, dtype=torch.float32)
    got = ttnn.to_torch(out)
    assert torch.equal(got.view(torch.int32), expected.view(torch.int32)), f"plan {plan['rect']} wrong"
