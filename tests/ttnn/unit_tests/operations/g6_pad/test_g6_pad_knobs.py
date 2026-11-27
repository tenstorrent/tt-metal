# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of every non-default block-knob value (g6_pad.KNOBS).

The default plan runs one block per core on every INPUTS shape, so the multi-block path
(block_tiles < per-core range, cb_input_tiles at depth 2, ring wrap) and the single-lane / coarse
band paths are reached only here. Each case turns one or more knobs, then checks bit equality with a
poisoned input padding. Also usable for A/B timing under run_safe_pytest.sh --profile.
"""

import sys

import pytest
import torch

from ttnn.operations.g6_pad import g6_pad as _g6_pad_fn

from .test_g6_pad import run_case

g6 = sys.modules[_g6_pad_fn.__module__]

KNOB_SETS = {
    "lanes1": {"band_lanes": 1},
    "band_block2": {"band_block_tiles": 2},
    "min_units32": {"min_units_per_core": 32},
    "block1_depth2": {"block_tiles": 1, "input_depth_blocks": 2},
    "block3_depth2": {"block_tiles": 3, "input_depth_blocks": 2},
    "block2_depth1": {"block_tiles": 2, "input_depth_blocks": 1},
}

SHAPES = [
    pytest.param((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 24), (0, 0)), 0.0, id="p4"),
    pytest.param((16, 1, 1536), ((0, 0), (0, 31), (0, 0)), 3.0, id="16x1x1536"),
    pytest.param((1, 1, 196, 196), ((0, 60), (0, 60)), -1.7014118346046923e38, id="hw_196"),
    pytest.param((1, 185, 128), ((0, 71), (0, 0)), 0, id="h_185x128"),
]


@pytest.mark.parametrize("knobs", list(KNOB_SETS), ids=list(KNOB_SETS))
@pytest.mark.parametrize("shape, padding, value", SHAPES)
def test_g6_pad_knobs(device, monkeypatch, shape, padding, value, knobs):
    monkeypatch.setattr(g6, "KNOBS", {**g6.KNOBS, **KNOB_SETS[knobs]})
    torch.manual_seed(0)
    run_case(device, shape, padding, value, poison=1000.0)
