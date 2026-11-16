# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the x-broadcast knob settings that are not the shipped defaults.

Each case overrides program-descriptor knobs (protocol, round size, injector placement, compute K-block)
and checks the output plus bit-identity across two calls.
"""

import pytest
import torch
import ttnn

import ttnn.operations.wq_b_linear.wq_b_linear_program_descriptor as pd
from eval.golden_tests.wq_b_linear.helpers import TOLERANCES, WEIGHT_SCALE
from eval.metrics import Tolerance, check_output
from ttnn.operations.wq_b_linear import wq_b_linear

KNOB_CASES = {
    "per_round_8_rounds": {"K_BLOCK_TILES": 6, "X_ROUND_K_BLOCKS": 1},
    "per_round_1_round": {"X_ROUND_K_BLOCKS": 16},
    "per_round_capped_to_8_channels": {"K_BLOCK_TILES": 3, "X_ROUND_K_BLOCKS": 1},
    "per_round_last_row": {"X_INJECTOR_PLACEMENT": "last_row"},
    "per_round_concurrent": {"X_ROUNDS_CONCURRENT": True},
    "rotating_chained": {"X_MCAST_CHANNELS": "rotating", "K_BLOCK_TILES": 6, "X_ROUND_K_BLOCKS": 1},
    "rotating_concurrent": {"X_MCAST_CHANNELS": "rotating", "X_ROUNDS_CONCURRENT": True},
}


@pytest.mark.parametrize("knobs", list(KNOB_CASES))
def test_x_mcast_knobs(device, monkeypatch, knobs):
    for name, value in KNOB_CASES[knobs].items():
        monkeypatch.setattr(pd, name, value)
    torch.manual_seed(7)
    x = torch.randn(1, 1, 160, 1536).to(torch.bfloat16)
    w = (torch.randn(1536, 4096) * WEIGHT_SCALE).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    expected = torch.matmul(x.to(torch.float64), ttnn.to_torch(tw).to(torch.float64))

    y = wq_b_linear(tx, tw)
    check_output(
        y,
        expected,
        shape=[1, 1, 160, 4096],
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=Tolerance(pcc=TOLERANCES[(ttnn.bfloat8_b, False)]),
    )
    assert torch.equal(ttnn.to_torch(wq_b_linear(tx, tw)), ttnn.to_torch(y))
