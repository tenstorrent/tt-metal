# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_reshape, Phase 0 (bfloat16, TILE_LAYOUT, DRAM interleaved).

IMMUTABLE SPEC: the implementer must not modify this file.

g6_reshape is a pure move: every logical output value must equal the input value at the same
row-major linear position bit for bit (helpers.TOLERANCES is ("exact", None) for every dtype),
so the gate is bit equality of every logical value, never PCC.

Case selection (BB-22): the bfloat16 TILE production cases of feature_spec.INPUTS, each pinned
to the regime op_design.md assigns it; gather_general pinned with production geometries of the
int32 / float32 cases run in the Phase 0 dtype; one scatter_aligned shape whose lanes write two
halves of one DRAM granule; the spec's rules (poisoned padding, special values, -1, tuple shape,
cache key, round trip, validation errors); and the multi-block / staging-flush / override /
single-lane paths, reached on production shapes by turning the knobs op_design.md names in
g6_reshape_program_descriptor.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn

from ttnn.operations.g6_reshape import g6_reshape, g6_reshape_plan
import ttnn.operations.g6_reshape.g6_reshape_program_descriptor as g6_desc


GATHER_ALIGNED = "gather_aligned"
SCATTER_ALIGNED = "scatter_aligned"
GATHER_GENERAL = "gather_general"


# ---------------------------------------------------------------------------
# Reference and helpers
# ---------------------------------------------------------------------------


def torch_reference(x: torch.Tensor, shape) -> torch.Tensor:
    return torch.reshape(x, tuple(int(n) for n in shape))


def assert_bit_exact(got: torch.Tensor, expected: torch.Tensor):
    assert list(got.shape) == list(expected.shape), f"shape {list(got.shape)} != {list(expected.shape)}"
    bad = (got.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)).reshape(-1)
    n_bad = int(bad.sum())
    assert n_bad == 0, f"{n_bad} of {bad.numel()} logical values differ; first at linear index {int(bad.nonzero()[0])}"


def to_device(x: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )


def random_input(shape, seed=42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def run_and_check(x_torch: torch.Tensor, ttnn_input: ttnn.Tensor, shape):
    out = g6_reshape(ttnn_input, shape)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config() == ttnn_input.memory_config()
    got = ttnn.to_torch(out)
    assert_bit_exact(got, torch_reference(x_torch, shape))
    return got


# With MIN_STRIPS_PER_LANE tuned as high as the grid-synchronization lamp measures (16) and two
# lanes per core, a shape with this many strips still needs more than one core.
MULTI_CORE_MIN_WORK_UNITS = 128


def check_plan(ttnn_input: ttnn.Tensor, shape, expected_regime):
    plan = g6_reshape_plan(ttnn_input, shape)
    assert plan["regime"] == expected_regime, f"regime {plan['regime']!r}, design says {expected_regime!r}"
    assert plan["num_core_ranges"] <= 2, f"{plan['num_core_ranges']} core ranges (BB-17)"
    expected_lanes = min(
        plan["lanes_per_core"] * plan["grid_cores"],
        math.ceil(plan["work_units"] / plan["min_strips_per_lane"]),
    )
    assert plan["n_lanes"] == expected_lanes, f"n_lanes {plan['n_lanes']} is not the blocking knobs' choice"
    assert plan["n_cores"] == math.ceil(plan["n_lanes"] / plan["lanes_per_core"])
    if plan["work_units"] >= MULTI_CORE_MIN_WORK_UNITS:
        assert plan["n_cores"] > 1, f"{plan['work_units']} work units ran on one core"
    return plan


def _id(case):
    in_shape, out_shape = case[0], case[1]
    return "x".join(map(str, in_shape)) + "-to-" + "x".join(map(str, out_shape))


# ---------------------------------------------------------------------------
# Production bfloat16 TILE cases, each pinned to its regime
# ---------------------------------------------------------------------------

PRODUCTION_BF16_TILE = [
    ((1, 16, 6, 256), [1, 1, 16, 1536], GATHER_ALIGNED),  # Qwen #349
    ((16, 1, 64), [1, 1, 16, 64], GATHER_ALIGNED),  # Qwen #353
    ((16, 1, 1536), [1, 1, 16, 1536], GATHER_ALIGNED),  # Qwen #357
    ((1, 1, 16, 4608), [16, 1, 4608], SCATTER_ALIGNED),  # Qwen #364
    ((1, 16, 64), [16, 1, 64], SCATTER_ALIGNED),  # Qwen #367
    ((16, 1, 5120), [1, 1, 16, 5120], GATHER_ALIGNED),  # Qwen #370
    ((2, 1, 5120), [1, 1, 2, 5120], GATHER_ALIGNED),  # Qwen #382
    ((1, 8, 6, 256), [1, 1, 8, 1536], GATHER_ALIGNED),  # Qwen #490
    ((8, 1, 64), [1, 1, 8, 64], GATHER_ALIGNED),  # Qwen #494
    ((8, 1, 1536), [1, 1, 8, 1536], GATHER_ALIGNED),  # Qwen #498
    ((1, 1, 8, 4608), [8, 1, 4608], SCATTER_ALIGNED),  # Qwen #505
    ((1, 8, 64), [8, 1, 64], SCATTER_ALIGNED),  # Qwen #508
    ((8, 1, 5120), [1, 1, 8, 5120], GATHER_ALIGNED),  # Qwen #511
    ((1, 1, 6, 256), [1, 1, 1536], SCATTER_ALIGNED),  # Qwen #553
]


@pytest.mark.parametrize("case", PRODUCTION_BF16_TILE, ids=[_id(c) for c in PRODUCTION_BF16_TILE])
def test_production_bf16_tile(case, device):
    in_shape, out_shape, regime = case
    x = random_input(in_shape)
    tt_x = to_device(x, device)
    check_plan(tt_x, out_shape, regime)
    run_and_check(x, tt_x, out_shape)


# ---------------------------------------------------------------------------
# gather_general (a last dim not a multiple of 16): production geometries in bf16
# ---------------------------------------------------------------------------

GENERAL_BF16_TILE = [
    ((1, 185), [1, 1, 185, 1]),  # FLUX #168 geometry: one row -> one column
    ((1, 77), [77, 1]),  # Qwen #456 geometry
    ((1, 4096), [4096, 1]),  # Qwen #529 geometry (flagged p1): 128 output tiles over the grid
    ((8,), [8, 1]),  # Qwen #400 geometry: rank 1 -> rank 2
    ((8, 1), [8]),  # Qwen #399 geometry: rank 2 -> rank 1, one output strip
]


@pytest.mark.parametrize("case", GENERAL_BF16_TILE, ids=[_id(c) for c in GENERAL_BF16_TILE])
def test_general_bf16_tile(case, device):
    in_shape, out_shape = case
    x = random_input(in_shape)
    tt_x = to_device(x, device)
    check_plan(tt_x, out_shape, GATHER_GENERAL)
    run_and_check(x, tt_x, out_shape)


# ---------------------------------------------------------------------------
# scatter_aligned map-side writes of two lanes into one 64-B DRAM granule
# ---------------------------------------------------------------------------


def test_scatter_lanes_share_output_granule(device):
    # Output image 1 rows 0 and 1 come from input rows 3 and 4, i.e. two different strips.
    in_shape, out_shape = (1, 1, 30, 64), [10, 3, 64]
    x = random_input(in_shape)
    tt_x = to_device(x, device)
    plan = check_plan(tt_x, out_shape, SCATTER_ALIGNED)
    assert plan["n_lanes"] > 2
    run_and_check(x, tt_x, out_shape)


# ---------------------------------------------------------------------------
# Rule 2: input tile padding (NaN) never reaches a logical output position
# ---------------------------------------------------------------------------

POISON_CASES = [
    ((16, 1, 64), [1, 1, 16, 64]),  # gather_aligned
    ((1, 16, 6, 256), [1, 1, 16, 1536]),  # gather_aligned
    ((1, 1, 16, 4608), [16, 1, 4608]),  # scatter_aligned
    ((1, 185), [1, 1, 185, 1]),  # gather_general
]


@pytest.mark.parametrize("case", POISON_CASES, ids=[_id(c) for c in POISON_CASES])
def test_poisoned_padding(case, device):
    in_shape, out_shape = case
    x = random_input(in_shape)
    tt_x = ttnn.fill_implicit_tile_padding(to_device(x, device), float("nan"))
    run_and_check(x, tt_x, out_shape)


# ---------------------------------------------------------------------------
# Rule 1: bit exactness on special values (NaN, +-inf, -0.0, denormal, last mantissa bit)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        ((1, 16, 6, 256), [1, 1, 16, 1536]),
        ((1, 1, 16, 4608), [16, 1, 4608]),
        ((1, 185), [1, 1, 185, 1]),
    ],
    ids=["gather_aligned", "scatter_aligned", "gather_general"],
)
def test_special_values(case, device):
    in_shape, out_shape = case
    torch.manual_seed(42)
    flat = torch.randn(in_shape, dtype=torch.float32).to(torch.bfloat16).reshape(-1)
    flat = (flat.view(torch.int16) | 1).view(torch.bfloat16).clone()
    specials = [float("nan"), float("inf"), float("-inf"), -0.0, 9.18e-41, 3.0e38]
    stride = max(flat.numel() // len(specials), 1)
    for i, v in enumerate(specials):
        flat[min(i * stride, flat.numel() - 1)] = v
    x = flat.reshape(in_shape)
    run_and_check(x, to_device(x, device), out_shape)


# ---------------------------------------------------------------------------
# Signature: -1 entry, tuple shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [((16, 1, 1536), [1, 1, -1, 1536]), ((1, 4096), [-1, 1])],
    ids=["inferred_rows", "inferred_column"],
)
def test_inferred_entry(case, device):
    in_shape, shape = case
    x = random_input(in_shape)
    got = run_and_check(x, to_device(x, device), shape)
    assert list(got.shape) == list(torch.reshape(x, shape).shape)


def test_tuple_shape(device):
    x = random_input((1, 1, 16, 4608))
    run_and_check(x, to_device(x, device), (16, 1, 4608))


# ---------------------------------------------------------------------------
# BB-07: the program description cache never replays a wrong map or a stale address
# ---------------------------------------------------------------------------


def test_two_targets_same_input(device):
    x = random_input((16, 1, 64))
    tt_x = to_device(x, device)
    run_and_check(x, tt_x, [1, 1, 16, 64])
    run_and_check(x, tt_x, [1, 1, 1, 1024])
    run_and_check(x, tt_x, [1, 1, 16, 64])


def test_same_signature_fresh_tensors(device):
    in_shape, out_shape = (16, 1, 5120), [1, 1, 16, 5120]
    first = random_input(in_shape, seed=42)
    run_and_check(first, to_device(first, device), out_shape)
    other = random_input((1, 1, 16, 4608), seed=43)
    run_and_check(other, to_device(other, device), [16, 1, 4608])
    second = random_input(in_shape, seed=44)
    run_and_check(second, to_device(second, device), out_shape)


# ---------------------------------------------------------------------------
# Round trip: g6_reshape(g6_reshape(x, e), d) == x bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [((1, 1, 16, 4608), [16, 1, 4608]), ((1, 185), [1, 1, 185, 1])],
    ids=["head_regroup", "row_to_column"],
)
def test_round_trip(case, device):
    in_shape, out_shape = case
    x = random_input(in_shape)
    y = g6_reshape(to_device(x, device), out_shape)
    back = g6_reshape(y, list(in_shape))
    assert_bit_exact(ttnn.to_torch(back), x)


# ---------------------------------------------------------------------------
# Knob paths on production shapes: multi-block lanes, slot reuse, staging flushes,
# forced regimes, one lane per core
# ---------------------------------------------------------------------------

MULTI_BLOCK_CASES = [
    ((16, 1, 5120), [1, 1, 16, 5120], GATHER_ALIGNED),
    ((1, 1, 16, 4608), [16, 1, 4608], SCATTER_ALIGNED),
    ((1, 4096), [4096, 1], GATHER_GENERAL),
]


@pytest.mark.parametrize("slot_depth", [1, 2], ids=["depth1", "depth2"])
@pytest.mark.parametrize("case", MULTI_BLOCK_CASES, ids=[_id(c) for c in MULTI_BLOCK_CASES])
def test_multi_block_knobs(case, slot_depth, device, monkeypatch):
    in_shape, out_shape, regime = case
    monkeypatch.setattr(g6_desc, "BLOCK_SLOT_BYTES_CAP", 2048)
    monkeypatch.setattr(g6_desc, "STAGING_SLOTS_CAP", 4)
    monkeypatch.setattr(g6_desc, "COPY_ENTRIES_CAP", 4)
    monkeypatch.setattr(g6_desc, "MIN_STRIPS_PER_LANE", 64)
    monkeypatch.setattr(g6_desc, "SLOT_DEPTH", slot_depth)
    x = random_input(in_shape)
    tt_x = to_device(x, device)
    plan = check_plan(tt_x, out_shape, regime)
    assert plan["max_blocks_per_lane"] > 1, "the knob override did not produce multi-block lanes"
    assert plan["slot_depth"] == slot_depth
    run_and_check(x, tt_x, out_shape)


@pytest.mark.parametrize("regime", [GATHER_ALIGNED, SCATTER_ALIGNED, GATHER_GENERAL])
def test_regime_override(regime, device, monkeypatch):
    in_shape, out_shape = (1, 16, 6, 256), [1, 1, 16, 1536]
    monkeypatch.setattr(g6_desc, "REGIME_OVERRIDE", regime)
    x = random_input(in_shape)
    tt_x = to_device(x, device)
    check_plan(tt_x, out_shape, regime)
    run_and_check(x, tt_x, out_shape)


def test_single_lane_per_core(device, monkeypatch):
    in_shape, out_shape = (16, 1, 5120), [1, 1, 16, 5120]
    monkeypatch.setattr(g6_desc, "LANES_PER_CORE", 1)
    x = random_input(in_shape)
    tt_x = to_device(x, device)
    plan = check_plan(tt_x, out_shape, GATHER_ALIGNED)
    assert plan["lanes_per_core"] == 1
    run_and_check(x, tt_x, out_shape)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_shape",
    [[3844, 2], [0, 3844], [-1, -1], [-2, 3844], [1, 1, 1, 1, 3844], [-1, 7], []],
    ids=["product", "zero", "two_inferred", "below_minus_one", "rank5", "inferred_not_divisible", "rank0"],
)
def test_validation_errors(bad_shape, device, expect_error):
    x = random_input((1, 3844))
    with expect_error((ValueError, RuntimeError), ""):
        g6_reshape(to_device(x, device), bad_shape)


def test_input_rank_above_four_refused(device, expect_error):
    x = random_input((1, 1, 1, 2, 32))
    tt_x = to_device(x, device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_reshape(tt_x, [64])
