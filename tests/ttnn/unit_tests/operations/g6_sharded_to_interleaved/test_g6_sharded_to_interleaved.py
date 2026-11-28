# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_sharded_to_interleaved (immutable spec — do not edit).

Scope (BB-22): the two production cases of the golden INPUTS and the cases the
spec's rules require — another shard spec per case (column-major, short last
shard), the destination taken from memory_config, repeat, and input intact.
The gate is exact (bit equality); TOLERANCES for this op are ("exact", None).
Cells outside the op's current SUPPORTED raise a SupportRefusal → xfail.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.g6_sharded_to_interleaved import g6_sharded_to_interleaved

TILE = 32

_P2 = dict(
    shape=(1, 5120),
    dtype=ttnn.bfloat16,
    scheme="width",
    shard=(32, 160),
    grid=(((0, 0), (7, 3)),),
    orientation="row_major",
    buffer="dram",
)
_P1 = dict(
    shape=(1, 128, 256, 32),
    dtype=ttnn.bfloat8_b,
    scheme="height",
    shard=(320, 32),
    grid=(((0, 0), (10, 8)), ((0, 9), (3, 9))),
    orientation="row_major",
    buffer="l1",
)

CASES = {
    "p2_flux_width_bf16_dram": _P2,
    "p2_other_spec_colmajor_short_last": {
        **_P2,
        "shard": (32, 96),
        "grid": (((0, 0), (8, 5)),),
        "orientation": "column_major",
    },
    "p2_flux_to_l1": {**_P2, "buffer": "l1"},
    "p1_panoptic_height_bf8b_l1": _P1,
    "p1_other_spec_colmajor_short_last": {
        **_P1,
        "shard": (352, 32),
        "grid": (((0, 0), (8, 9)), ((9, 0), (9, 3))),
        "orientation": "column_major",
    },
}

_SCHEME = {"width": ttnn.TensorMemoryLayout.WIDTH_SHARDED, "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED}
_ORIENT = {"row_major": ttnn.ShardOrientation.ROW_MAJOR, "column_major": ttnn.ShardOrientation.COL_MAJOR}
_OUT_MC = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
_BUF = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1}


def torch_reference(x):
    return torch.clone(x)


def _make_input(shape):
    """randn with a distinct power-of-two scale per tile, so a wrong page map is visible."""
    torch.manual_seed(42)
    rows, cols = math.prod(shape[:-1]), shape[-1]
    x = torch.randn(rows, cols, dtype=torch.float32)
    tile_cols = -(-cols // TILE)
    tile_idx = (torch.arange(rows) // TILE)[:, None] * tile_cols + (torch.arange(cols) // TILE)[None, :]
    x = x * torch.pow(2.0, (tile_idx % 16 - 8).to(torch.float32))
    return x.reshape(shape).to(torch.bfloat16)


def _input_mc(device, case):
    size = device.compute_with_storage_grid_size()
    for _, (x1, y1) in case["grid"]:
        if x1 >= size.x or y1 >= size.y:
            pytest.skip(f"shard grid end ({x1},{y1}) outside worker grid ({size.x},{size.y})")
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in case["grid"]]
    )
    spec = ttnn.ShardSpec(grid, case["shard"], _ORIENT[case["orientation"]])
    return ttnn.MemoryConfig(_SCHEME[case["scheme"]], ttnn.BufferType.L1, spec)


_BITS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _assert_bits_equal(actual, expected, what):
    assert tuple(actual.shape) == tuple(
        expected.shape
    ), f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    # bfloat16 comes back as torch.bfloat16; bfloat8_b decodes to torch.float32. Compare the raw bits
    # at whichever width came back (bf8b decode is a pure function of the page bytes).
    assert actual.dtype == expected.dtype and actual.dtype in _BITS, f"{what}: dtype {actual.dtype} vs {expected.dtype}"
    bits = _BITS[actual.dtype]
    differ = actual.contiguous().view(bits) != expected.contiguous().view(bits)
    assert int(differ.sum()) == 0, f"{what}: {int(differ.sum())} of {differ.numel()} elements differ"


def _call(tt_input, out_mc):
    try:
        return g6_sharded_to_interleaved(tt_input, out_mc)
    except SupportRefusal as e:
        ttnn.deallocate(tt_input)
        pytest.xfail(f"outside current SUPPORTED: {e}")


@pytest.mark.parametrize("case_id", list(CASES))
def test_g6_sharded_to_interleaved(device, case_id):
    case = CASES[case_id]
    shape, dtype = case["shape"], case["dtype"]
    out_mc = _OUT_MC[case["buffer"]]

    torch_input = _make_input(shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_input_mc(device, case)
    )
    # bfloat16 is stored as given; bfloat8_b as the host packed it.
    source = torch_input if dtype == ttnn.bfloat16 else ttnn.to_torch(tt_input)
    expected = torch_reference(source)
    in_mc_before = tt_input.memory_config()
    in_addr = tt_input.buffer_address()

    out = _call(tt_input, out_mc)

    assert tuple(out.shape) == tuple(shape)
    assert out.dtype == dtype
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.memory_config().buffer_type == _BUF[case["buffer"]]
    assert out.buffer_address() != in_addr or case["buffer"] == "dram", "output must be new storage"
    out_torch = ttnn.to_torch(out)
    _assert_bits_equal(out_torch, expected, "output")

    # Repeat: bit-identical result in a new buffer (BB-06).
    out2 = g6_sharded_to_interleaved(tt_input, out_mc)
    assert out2.buffer_address() != out.buffer_address()
    _assert_bits_equal(ttnn.to_torch(out2), out_torch, "repeat")

    # Input intact: still allocated, same placement and address, same values.
    assert tt_input.is_allocated()
    assert tt_input.buffer_address() == in_addr
    assert tt_input.memory_config() == in_mc_before
    _assert_bits_equal(ttnn.to_torch(tt_input), source, "input")

    for t in (out, out2, tt_input):
        ttnn.deallocate(t)


# Refinement 2: the L1-output plan (dual_noc + bank_stagger) is selected only for an L1 output with
# enough pages to congest one NoC (p1: 1024 pages); p2 (160 pages, DRAM or L1) keeps the Phase 0 plan.
_EXPECT_L1_PLAN = {
    "p2_flux_width_bf16_dram": False,
    "p2_flux_to_l1": False,
    "p1_panoptic_height_bf8b_l1": True,
}


@pytest.mark.parametrize("case_id", list(_EXPECT_L1_PLAN))
def test_g6_sharded_to_interleaved_plan_selection(device, case_id):
    from ttnn.operations.g6_sharded_to_interleaved import last_plan

    case = CASES[case_id]
    tt_input = ttnn.from_torch(
        _make_input(case["shape"]),
        dtype=case["dtype"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_input_mc(device, case),
    )
    out = g6_sharded_to_interleaved(tt_input, _OUT_MC[case["buffer"]])
    plan = last_plan()
    want = _EXPECT_L1_PLAN[case_id]
    assert plan["dual_noc"] is want and plan["bank_stagger"] is want, plan
    assert plan["writer_lanes"] == 1
    for t in (out, tt_input):
        ttnn.deallocate(t)


@pytest.mark.parametrize(
    "bad",
    ["interleaved_input", "interleaved_mc_required"],
)
def test_g6_sharded_to_interleaved_validation(device, bad, expect_error):
    case = CASES["p2_flux_width_bf16_dram"]
    torch_input = _make_input(case["shape"])
    if bad == "interleaved_input":
        x = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_mc = ttnn.DRAM_MEMORY_CONFIG
    else:
        mc = _input_mc(device, case)
        x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        out_mc = mc  # sharded output memory config is not allowed
    with expect_error((ValueError, RuntimeError), ""):
        g6_sharded_to_interleaved(x, out_mc)
    ttnn.deallocate(x)
