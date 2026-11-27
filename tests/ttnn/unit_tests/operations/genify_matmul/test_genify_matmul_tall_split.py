# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 19: tall, narrow interleaved matmuls (PDL 1x1 convs read in place) on an M-only split over the grid.

Pins what the golden suite does not see:
  - the plan: a tall, narrow shape takes an M-only split over most of the grid with B resident, its B (+ bias row)
    multicast once per rectangle of the split (R1_B_MCAST) and the A in-flight cap (R1_A_INFLIGHT_TILES);
  - the walk's numerics under every epilogue family (the row rides in B's delivery and is kept across M-blocks),
    with A in DRAM and in L1, a ragged M, and a split whose last rectangle is a single core (it reads its own B);
  - the parked / alternate settings stay correct: the distributed B delivery (R1_B_DISTRIBUTED), the cap off, the
    multicast off.
Accuracy gate: helpers.TOLERANCES / LOW_FIDELITY_TOLERANCES via run_genify_matmul.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor, run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

_BF16, _BFP8 = ttnn.bfloat16, ttnn.bfloat8_b
_LOFI = ttnn.MathFidelity.LoFi
_L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


def _plan(device, a, b, *, a_mc, o_mc, epilogue="bias_relu"):
    ta = create_ttnn_input_tensor(torch.randn(*a), device, dtype=_BFP8, layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(torch.randn(*b), device, dtype=_BFP8, layout=ttnn.TILE_LAYOUT)
    row = create_ttnn_input_tensor(torch.randn(1, b[-1]), device, dtype=_BFP8, layout=ttnn.TILE_LAYOUT)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a[:-1]) + [b[-1]]), _BFP8, ttnn.TILE_LAYOUT, device, o_mc)
    geom, grid, blk, _ = pd.plan(ta, tb, out, True, device, epilogue=epilogue, row=row, math_fidelity=_LOFI)
    for t in (ta, tb, row, out):
        ttnn.deallocate(t)
    return geom, grid, blk


@pytest.mark.parametrize(
    "a, b, a_mc",
    [
        ((1, 1, 32768, 256), (256, 32), _L1),  # decoder res2 projection
        ((1, 1, 32768, 256), (256, 128), _DRAM),  # res3.0 conv1
    ],
    ids=["dec_res2", "res3_conv1"],
)
def test_tall_split_plan(device, a, b, a_mc):
    """The tall PDL convs: an M-only split over >= 64 cores, B resident and multicast, the A cap on."""
    geom, grid, blk = _plan(device, a, b, a_mc=a_mc, o_mc=_L1)
    cores = grid.grid_b * grid.grid_m * grid.grid_n
    assert grid.grid_n == 1 and grid.grid_b == 1 and cores >= pd.R1_A_INFLIGHT_MIN_CORES, grid
    assert blk.weight_resident and blk.num_k_blocks == 1, blk
    assert pd.r1_b_mcast(geom, grid, blk)
    assert pd.r1_a_inflight(geom, grid, blk) == pd.R1_A_INFLIGHT_TILES


def _run(device, a, b, *, dtype=_BFP8, epilogue="bias_relu", **extras):
    extras.setdefault("math_fidelity", _LOFI)
    run_genify_matmul(
        (a, b),
        dtype=dtype,
        weight_dtype=_BFP8,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        placement="interleaved",
        epilogue=epilogue,
        device=device,
        extras=extras,
    )


@pytest.mark.parametrize(
    "a, b, dtype, epilogue, extras",
    [
        ((1, 1, 32768, 256), (256, 32), _BFP8, "bias_relu", dict(act_buffer="l1", out_buffer="l1")),
        ((1, 1, 8192, 512), (512, 64), _BFP8, "bias_relu", dict(act_buffer="l1", out_buffer="l1")),
        ((1, 1, 8192, 512), (512, 256), _BFP8, "bias", dict(out_buffer="l1")),
        ((1, 1, 8192, 256), (256, 512), _BF16, "bias", dict(act_buffer="l1", out_buffer="l1", output_dtype=_BFP8)),
        ((1, 1, 8000, 256), (256, 64), _BF16, "none", {}),  # ragged M, no row
        ((1, 1, 8192, 256), (256, 64), _BF16, "addcmul", {}),  # the gate row, kept across M-blocks
    ],
    ids=["dec_res2", "dec_res3", "res4_conv1", "res3_short", "ragged_none", "addcmul"],
)
def test_tall_split_numerics(device, a, b, dtype, epilogue, extras):
    _run(device, a, b, dtype=dtype, epilogue=epilogue, **extras)


@pytest.mark.parametrize(
    "knobs",
    [
        dict(R1_B_DISTRIBUTED=True),
        dict(R1_A_INFLIGHT_TILES=0),
        dict(R1_B_MCAST=False),
        dict(R1_A_STAGGER=False, R1_A_INFLIGHT_TILES=1),
    ],
    ids=["distributed_b", "no_cap", "no_b_mcast", "cap1_no_stagger"],
)
def test_tall_split_knobs(device, monkeypatch, knobs):
    for name, value in knobs.items():
        monkeypatch.setattr(pd, name, value)
    _run(device, (1, 1, 32768, 256), (256, 128), act_buffer="l1", out_buffer="l1")


def test_tall_split_single_core_group(device, monkeypatch):
    """34 tile rows on 34 cores (one row each): rectangles 11 x 3 + a lone core, which reads its own B and row."""
    orig = pd.choose_split

    def forced(geom, gx, gy, budget):
        for grid in pd.grid_candidates(geom, gx, gy):
            if (grid.grid_b, grid.grid_m, grid.grid_n, grid.placement) == (1, 34, 1, "flat"):
                return grid, pd.choose_blocking(geom, grid, budget)
        return orig(geom, gx, gy, budget)

    monkeypatch.setattr(pd, "choose_split", forced)
    monkeypatch.setattr(pd, "R1_A_INFLIGHT_MIN_CORES", 1)
    _run(device, (1, 1, 34 * 32, 512), (512, 96), act_buffer="l1", out_buffer="l1")
