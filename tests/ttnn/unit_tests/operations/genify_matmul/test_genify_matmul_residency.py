# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 8: operand residency on one-K-block R1 / R1a walks, the L1-discounted split ranking, and the
output write-batching knob.

Pins what the golden suite does not see:
  - which residency a short-K shape takes (`act_resident`: A held across N-blocks, B streamed;
    `weight_resident`: B held across M-blocks, A streamed) and that the split keeps the R1a multicast;
  - PERF FOCUS 3's split (32 x 3 x 1: A read from DRAM once) vs the L1_READ_COST = 1.0 ranking
    (32 x 1 x 3, the Refinement 7 split, which now takes weight residency);
  - residency under every epilogue family (the epilogue path pops the retained operand itself);
  - the parked write-batch knob turned on (several row groups per writer wait / flush / pop).
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

# PERF FOCUS 3: PDL bilinear upsample, H pass (batched B in L1, bf8b, LoFi, 16-bit DEST).
_PF3 = dict(a=(1, 32, 512, 128), b=(1, 32, 128, 1024), dtype=_BFP8, wdtype=_BFP8, acc=False, b_mc=_L1)


def _plan(device, a, b, *, dtype=_BF16, wdtype=_BF16, acc=True, a_mc=_DRAM, b_mc=_DRAM, o_mc=_DRAM, epilogue="none"):
    ta = create_ttnn_input_tensor(torch.randn(*a), device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(torch.randn(*b), device, dtype=wdtype, layout=ttnn.TILE_LAYOUT, memory_config=b_mc)
    n_out = b[-1] // 2 if epilogue == "swiglu" else b[-1]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a[:-1]) + [n_out]), dtype, ttnn.TILE_LAYOUT, device, o_mc)
    geom, grid, blk, _ = pd.plan(ta, tb, out, acc, device, epilogue=epilogue)
    mcast = pd.uses_operand_mcast(geom, grid, blk)
    for t in (ta, tb, out):
        ttnn.deallocate(t)
    return grid, blk, mcast


def _run(device, a, b, *, dtype=_BF16, wdtype=_BF16, acc=True, epilogue="none", **extras):
    run_genify_matmul(
        (a, b),
        dtype=dtype,
        weight_dtype=wdtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=acc,
        placement="interleaved",
        epilogue=epilogue,
        device=device,
        extras=extras,
    )


@pytest.mark.parametrize("l1_cost", [0.25, 1.0], ids=["l1_discount", "bytes_are_bytes"])
def test_perf_focus_3_split(device, monkeypatch, l1_cost):
    """PERF FOCUS 3: the discounted ranking reads A from DRAM once (32x3x1, A resident, 8-column B
    blocks); bytes-are-bytes keeps the Refinement 7 split (32x1x3) with B resident. Pinned under the legacy
    ranking (SPLIT_COST_MODEL = False) whose L1 discount it documents; the Refinement 11 cost model picks a
    32x2x1 split for the golden case (measured 66.9 vs 69.6 us, median of 4)."""
    monkeypatch.setattr(pd, "SPLIT_COST_MODEL", False)
    monkeypatch.setattr(pd, "L1_READ_COST", l1_cost)
    grid, blk, _ = _plan(device, _PF3["a"], _PF3["b"], dtype=_BFP8, wdtype=_BFP8, acc=False, b_mc=_L1)
    if l1_cost < 1.0:
        assert (grid.grid_b, grid.grid_m, grid.grid_n) == (32, 3, 1)
        assert blk.act_resident and not blk.weight_resident and blk.block_n_tiles == 8 and blk.sb_h * blk.sb_w == 8
    else:
        assert (grid.grid_b, grid.grid_m, grid.grid_n) == (32, 1, 3)
        assert blk.weight_resident and not blk.act_resident and blk.block_m_tiles < grid.core_m_tiles
    _run(
        device,
        _PF3["a"],
        _PF3["b"],
        dtype=_BFP8,
        wdtype=_BFP8,
        acc=False,
        math_fidelity=_LOFI,
        weight_buffer="l1",
    )


# (A, B, dtype, acc, epilogue, expected residency, extras). The 2048-row shapes split as R1a rectangles
# (line multicast), so the skipped deliveries of the resident operand are exercised on injectors and
# receivers alike.
CASES = {
    "weight_res_mcast_none": ((1, 1, 2048, 256), (256, 512), _BF16, True, "none", "weight", {}),
    "weight_res_mcast_bias_relu": ((1, 1, 2048, 256), (256, 512), _BF16, True, "bias_relu", "weight", {}),
    "weight_res_mcast_swiglu": ((1, 1, 2048, 256), (256, 1024), _BF16, True, "swiglu", None, {}),
    "weight_res_mcast_addcmul": ((1, 1, 2048, 256), (256, 512), _BF16, True, "addcmul", "weight", {}),
    "act_res_mcast_none": ((1, 2048, 128), (128, 3072), _BF16, True, "none", "act", {}),
    "act_res_mcast_bias": ((1, 2048, 128), (128, 3072), _BF16, True, "bias", "act", {}),
    "act_res_mcast_swiglu_bf8": ((1, 2048, 128), (128, 3072), _BFP8, False, "swiglu", None, {"math_fidelity": _LOFI}),
    # PDL upsample W pass: A and the output in L1 interleaved, bf8b, LoFi, 16-bit DEST.
    "weight_res_wpass_l1": (
        (1, 32, 128, 256),
        (256, 1024),
        _BFP8,
        False,
        "none",
        "weight",
        {"math_fidelity": _LOFI, "act_buffer": "l1", "out_buffer": "l1"},
    ),
    # Ragged M (m_non_aligned) under residency.
    "ragged_m_bias": ((1, 1000, 128), (128, 768), _BF16, True, "bias", None, {}),
}


@pytest.mark.parametrize("case", CASES.keys())
def test_residency(device, case):
    a, b, dtype, acc, epilogue, expect, extras = CASES[case]
    a_mc = _L1 if extras.get("act_buffer") == "l1" else _DRAM
    o_mc = _L1 if extras.get("out_buffer") == "l1" else _DRAM
    grid, blk, mcast = _plan(device, a, b, dtype=dtype, wdtype=dtype, acc=acc, a_mc=a_mc, o_mc=o_mc, epilogue=epilogue)
    if expect == "weight":
        assert blk.weight_resident and mcast, (grid, blk)
    elif expect == "act":
        assert blk.act_resident and mcast, (grid, blk)
    _run(device, a, b, dtype=dtype, wdtype=dtype, acc=acc, epilogue=epilogue, **extras)


def test_residency_off_matches_whole_region(device, monkeypatch):
    """RESIDENT_SPLIT_BLOCKS = 1 restores the whole-region walk (the Refinement 7 path) on a residency shape."""
    monkeypatch.setattr(pd, "RESIDENT_SPLIT_BLOCKS", 1)
    _, blk, _ = _plan(device, (1, 2048, 128), (128, 3072))
    assert not (blk.act_resident or blk.weight_resident)
    _run(device, (1, 2048, 128), (128, 3072))


@pytest.mark.parametrize("epilogue", ["none", "swiglu"])
def test_write_batch_knob(device, monkeypatch, epilogue):
    """The parked write-batch knob, turned on: several row groups per writer wait / flush / pop."""
    monkeypatch.setattr(pd, "OUT_WRITE_BATCH_ROW_GROUPS", 4)
    _, blk, _ = _plan(device, (1, 384, 1536), (1536, 1024), epilogue=epilogue)
    assert blk.write_batch > 1, blk
    _run(device, (1, 384, 1536), (1536, 1024), epilogue=epilogue)
