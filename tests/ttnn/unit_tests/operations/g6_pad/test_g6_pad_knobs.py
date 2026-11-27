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
    # pipeline_min_units=1 lifts the path gate so the multi-block ring runs on every shape here.
    "block1_depth2": {"block_tiles": 1, "input_depth_blocks": 2, "pipeline_min_units": 1},
    "block3_depth2": {"block_tiles": 3, "input_depth_blocks": 2, "pipeline_min_units": 1},
    "block2_depth1": {"block_tiles": 2, "input_depth_blocks": 1, "pipeline_min_units": 1},
    # Refinement 2 knobs: rotated (bank_stagger) walk wrapping across blocks, NoC-grown fill page, and
    # the Phase 0 schedule (whole range, in-order, whole-page RISC fill) kept reachable.
    "stagger_block3": {"block_tiles": 3, "bank_stagger": 1, "pipeline_min_units": 1, "fill_seed_bytes": 64},
    "phase0_schedule": {"block_tiles": 0, "bank_stagger": 0, "fill_seed_bytes": 4096},
    # Refinement 3 load-balance knobs: cost-weighted contiguous split, two-segment (source region + new
    # dim-0 region) interleaved walk, and the hoisted fill pass -- alone and combined with the ring.
    "split_cost": {"split_mode": 1},
    "split_cost_fill_heavy": {"split_mode": 1, "copy_cost": 1, "fill_cost": 3},
    "split_mix": {"split_mode": 2},
    "split_mix_block3": {"split_mode": 2, "block_tiles": 3, "pipeline_min_units": 1, "bank_stagger": 1},
    "fill_upfront": {"fill_upfront": 1, "block_tiles": 1, "pipeline_min_units": 1},
    "split_mix_fill_upfront": {"split_mode": 2, "fill_upfront": 1, "block_tiles": 2, "pipeline_min_units": 1},
}

SHAPES = [
    pytest.param((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 24), (0, 0)), 0.0, id="p4"),
    pytest.param((16, 1, 1536), ((0, 0), (0, 31), (0, 0)), 3.0, id="16x1x1536"),
    pytest.param((1, 1, 196, 196), ((0, 60), (0, 60)), -1.7014118346046923e38, id="hw_196"),
    pytest.param((1, 185, 128), ((0, 71), (0, 0)), 0, id="h_185x128"),
    # dim-0 growth (two split regions) with row and column edge patches in the source region.
    pytest.param((3, 2, 50, 70), ((0, 2), (0, 1), (0, 10), (0, 30)), -5.0, id="lead_hw_3x2x50x70"),
]


@pytest.mark.parametrize("knobs", list(KNOB_SETS), ids=list(KNOB_SETS))
@pytest.mark.parametrize("shape, padding, value", SHAPES)
def test_g6_pad_knobs(device, monkeypatch, shape, padding, value, knobs):
    monkeypatch.setattr(g6, "KNOBS", {**g6.KNOBS, **KNOB_SETS[knobs]})
    torch.manual_seed(0)
    run_case(device, shape, padding, value, poison=1000.0)
