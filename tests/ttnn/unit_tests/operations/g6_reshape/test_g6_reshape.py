# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_reshape, Phase 0 (bfloat16, TILE_LAYOUT, DRAM interleaved).

IMMUTABLE SPEC — the implementer must not modify this file.

g6_reshape is a pure move: every logical output value equals the input value at the same
row-major linear position, bit for bit (helpers.TOLERANCES is ("exact", None) for every dtype).
The gate is therefore bit equality of every logical value, never PCC.

Case selection (BB-22): the bfloat16 TILE production cases of feature_spec.INPUTS, plus the
cases the spec's rules and the design require:
  - regime R2 (`segment_gather_general`, a W not a multiple of 16) pinned with the production
    geometries of the int32 / float32 cases, run in the Phase 0 dtype;
  - poisoned (NaN) input tile padding, special values, a -1 entry, a tuple shape,
    two targets on one input (program cache), the round trip, and the validation errors.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.g6_reshape import g6_reshape


# ---------------------------------------------------------------------------
# Reference and helpers
# ---------------------------------------------------------------------------


def torch_reference(x: torch.Tensor, shape) -> torch.Tensor:
    return torch.reshape(x, tuple(int(n) for n in shape))


def _bits(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.bfloat16
    return t.contiguous().view(torch.int16)


def assert_bit_exact(got: torch.Tensor, expected: torch.Tensor):
    assert list(got.shape) == list(expected.shape), f"shape {list(got.shape)} != {list(expected.shape)}"
    bad = (_bits(got) != _bits(expected)).reshape(-1)
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


def random_input(shape) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def run_and_check(x_torch: torch.Tensor, ttnn_input: ttnn.Tensor, shape):
    out = g6_reshape(ttnn_input, shape)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config() == ttnn_input.memory_config()
    expected = torch_reference(x_torch, shape)
    got = ttnn.to_torch(out)
    assert_bit_exact(got, expected)
    return got


# ---------------------------------------------------------------------------
# Production bfloat16 TILE cases (regime R1: W_in % 16 == 0 and W_out % 16 == 0)
# ---------------------------------------------------------------------------

PRODUCTION_BF16_TILE = [
    ((1, 16, 6, 256), [1, 1, 16, 1536]),  # Qwen #349
    ((16, 1, 64), [1, 1, 16, 64]),  # Qwen #353
    ((16, 1, 1536), [1, 1, 16, 1536]),  # Qwen #357
    ((1, 1, 16, 4608), [16, 1, 4608]),  # Qwen #364
    ((1, 16, 64), [16, 1, 64]),  # Qwen #367
    ((16, 1, 5120), [1, 1, 16, 5120]),  # Qwen #370
    ((2, 1, 5120), [1, 1, 2, 5120]),  # Qwen #382
    ((1, 8, 6, 256), [1, 1, 8, 1536]),  # Qwen #490
    ((8, 1, 64), [1, 1, 8, 64]),  # Qwen #494
    ((8, 1, 1536), [1, 1, 8, 1536]),  # Qwen #498
    ((1, 1, 8, 4608), [8, 1, 4608]),  # Qwen #505
    ((1, 8, 64), [8, 1, 64]),  # Qwen #508
    ((8, 1, 5120), [1, 1, 8, 5120]),  # Qwen #511
    ((1, 1, 6, 256), [1, 1, 1536]),  # Qwen #553
]


def _id(case):
    in_shape, out_shape = case
    return "x".join(map(str, in_shape)) + "-to-" + "x".join(map(str, out_shape))


@pytest.mark.parametrize("case", PRODUCTION_BF16_TILE, ids=[_id(c) for c in PRODUCTION_BF16_TILE])
def test_production_bf16_tile(case, device):
    in_shape, out_shape = case
    x = random_input(in_shape)
    run_and_check(x, to_device(x, device), out_shape)


# ---------------------------------------------------------------------------
# Regime R2 pins (a W not a multiple of 16): production geometries in the Phase 0 dtype
# ---------------------------------------------------------------------------

GENERAL_SEGMENT_BF16_TILE = [
    ((1, 185), [1, 1, 185, 1]),  # FLUX #168 geometry: one row -> one column
    ((1, 207), [1, 1, 207, 1]),  # FLUX #172 geometry
    ((1, 77), [77, 1]),  # Qwen #456 geometry: row -> column
    ((1, 4096), [4096, 1]),  # Qwen #529 geometry (flagged p1), 128 output tiles
    ((8,), [8, 1]),  # Qwen #400 geometry: rank 1 -> rank 2
    ((8, 1), [8]),  # Qwen #399 geometry: rank 2 -> rank 1
]


@pytest.mark.parametrize("case", GENERAL_SEGMENT_BF16_TILE, ids=[_id(c) for c in GENERAL_SEGMENT_BF16_TILE])
def test_general_segment_bf16_tile(case, device):
    in_shape, out_shape = case
    x = random_input(in_shape)
    run_and_check(x, to_device(x, device), out_shape)


# ---------------------------------------------------------------------------
# Rule 2: input tile padding (NaN) never reaches a logical output position
# ---------------------------------------------------------------------------

POISON_CASES = [
    ((16, 1, 64), [1, 1, 16, 64]),
    ((1, 16, 6, 256), [1, 1, 16, 1536]),
    ((1, 1, 16, 4608), [16, 1, 4608]),
    ((1, 185), [1, 1, 185, 1]),
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


def test_special_values(device):
    in_shape, out_shape = (1, 16, 6, 256), [1, 1, 16, 1536]
    torch.manual_seed(42)
    flat = torch.randn(in_shape, dtype=torch.float32).to(torch.bfloat16).reshape(-1)
    flat = (flat.view(torch.int16) | 1).view(torch.bfloat16).clone()
    specials = [float("nan"), float("inf"), float("-inf"), -0.0, 9.18e-41, 3.0e38]
    stride = flat.numel() // len(specials)
    for i, v in enumerate(specials):
        flat[i * stride] = v
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
# BB-07: two targets on one input (the program cache must not replay the first map)
# ---------------------------------------------------------------------------


def test_two_targets_same_input(device):
    x = random_input((16, 1, 64))
    tt_x = to_device(x, device)
    run_and_check(x, tt_x, [1, 1, 16, 64])
    run_and_check(x, tt_x, [1, 1, 1, 1024])
    run_and_check(x, tt_x, [1, 1, 16, 64])


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
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_shape",
    [[3844, 2], [0, 3844], [-1, -1], [-2, 3844], [1, 1, 1, 1, 3844]],
    ids=["product", "zero", "two_inferred", "below_minus_one", "rank5"],
)
def test_validation_errors(bad_shape, device, expect_error):
    x = random_input((1, 3844))
    with expect_error((ValueError, RuntimeError), ""):  # any message
        g6_reshape(to_device(x, device), bad_shape)


def test_input_rank_above_four_refused(device, expect_error):
    x = random_input((1, 1, 1, 2, 32))
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), ""):  # any message
        g6_reshape(tt_x, [64])
