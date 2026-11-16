# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the W-stream / n-split knob settings that are not the shipped defaults.

Each case overrides program-descriptor knobs (core n share, placement, bank-coalesced columns, per-group read
depth, NoC1 share of the W read, the R4 K-remainder split and its slice / combine positions) and checks the output plus bit-identity across two calls.
"""

import pytest
import torch
import ttnn

import ttnn.operations.wq_b_linear.wq_b_linear_program_descriptor as pd
from eval.golden_tests.wq_b_linear.helpers import TOLERANCES, WEIGHT_SCALE
from eval.metrics import Tolerance, check_output
from ttnn.operations.wq_b_linear import wq_b_linear

KNOB_CASES = {
    "fill_grid_row_major": {"CORE_N_TILES": 0, "CORE_PLACEMENT": "row_major", "W_BANK_COALESCE": False},
    "fill_grid_spread": {"CORE_N_TILES": 0, "CORE_PLACEMENT": "spread"},
    "fill_grid_coalesced": {"CORE_N_TILES": 0, "W_BANK_COALESCE": True},
    "even_2_columns": {"CORE_N_TILES": 2, "W_BANK_COALESCE": False},
    "even_2_columns_coalesced_spread": {"CORE_N_TILES": 2, "W_BANK_COALESCE": True, "CORE_PLACEMENT": "spread"},
    "even_4_columns_coalesced": {"CORE_N_TILES": 4, "W_BANK_COALESCE": True},
    "g1_deeper_reads": {"CORE_N_TILES": 0, "W_READS_IN_FLIGHT": 2, "W_READS_IN_FLIGHT_G1": 4},
    "noc1_w_none": {"W_NOC1_K_BLOCKS": 0},
    "noc1_w_4_blocks_fill_grid": {"CORE_N_TILES": 0, "W_NOC1_K_BLOCKS": 4},
    "noc1_w_6_blocks_even_coalesced": {"CORE_N_TILES": 2, "W_BANK_COALESCE": True, "W_NOC1_K_BLOCKS": 6},
    "noc1_w_all_but_one": {"W_NOC1_K_BLOCKS": 64},
    "k_remainder": {"K_REMAINDER_SPLIT": True},
    "k_remainder_slices_and_combine_last": {
        "K_REMAINDER_SPLIT": True,
        "SLICE_EARLY_K_BLOCKS": 0,
        "COMBINE_EARLY_K_BLOCKS": 0,
    },
    "k_remainder_slices_first": {"K_REMAINDER_SPLIT": True, "SLICE_EARLY_K_BLOCKS": 64},
    "k_remainder_kb4": {"K_REMAINDER_SPLIT": True, "K_BLOCK_TILES": 4, "X_ROUND_K_BLOCKS": 3},
    "noc1_w_rotating_x": {
        "X_MCAST_CHANNELS": "rotating",
        "K_BLOCK_TILES": 6,
        "X_ROUND_K_BLOCKS": 1,
        "W_NOC1_K_BLOCKS": 2,
    },
}


@pytest.mark.parametrize("knobs", list(KNOB_CASES))
def test_w_stream_knobs(device, monkeypatch, knobs):
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
