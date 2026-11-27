# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for g6_typecast (Phase 0) — the immutable spec for the implementer.

Phase 0 SUPPORTED: bfloat16 TILE input (DRAM or L1 interleaved), ranks 3 and 4, any alignment,
destinations float32, bfloat8_b, int32. Cases are production cases of
eval/golden_tests/g6_typecast/feature_spec.py (BB-22), covering single-tile, multi-tile,
non-square, multi-batch, w/h-non-aligned and L1 placement.

Grading follows eval/golden_tests/g6_typecast/helpers.py:TOLERANCES: every destination except
bfloat8_b is bit-exact against torch.Tensor.to; bfloat8_b is within one quantization step
2^(e_max - 6) of each 16-value group against the host rendering (ttnn.from_torch bfloat8_b).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_typecast import g6_typecast

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def torch_reference(x, dtype):
    """torch.Tensor.to — the contract. bfloat8_b has no torch dtype: exact values, rendered later."""
    if dtype == ttnn.float32:
        return x.to(torch.float32)
    if dtype == ttnn.int32:
        return x.to(torch.int32)
    if dtype == ttnn.bfloat8_b:
        return x.to(torch.float32)
    raise ValueError(dtype)


def make_input(shape, dst):
    """Phase 0 input generator (bfloat16 source), seeded."""
    torch.manual_seed(42)
    n = 1
    for d in shape:
        n *= d
    if dst == ttnn.int32:
        # magnitudes up to 2^30 with fractions, plus truncation and full-width witnesses
        u = torch.rand(n) * 32 - 2
        sign = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
        x = (sign * torch.pow(2.0, u)).to(torch.bfloat16)
        witnesses = [-2.75, -0.5, 0.5, 2.75, 127.5, -(2.0**31 - 2.0**23), 2.0**30]
        k = min(len(witnesses), n)
        x[:k] = torch.tensor(witnesses[:k], dtype=torch.bfloat16)
    else:
        x = torch.randn(n).to(torch.bfloat16)
    return x.reshape(shape)


def check_exact(actual, expected, dst):
    if dst == ttnn.float32:
        a = actual.to(torch.float32).contiguous().view(torch.int32)
        e = expected.to(torch.float32).contiguous().view(torch.int32)
    else:
        a = actual.to(torch.int64) & 0xFFFFFFFF
        e = expected.to(torch.int64) & 0xFFFFFFFF
    bad = a != e
    assert int(bad.sum()) == 0, f"{int(bad.sum())}/{bad.numel()} values differ from torch.Tensor.to"


def _groups(t):
    t = t.to(torch.float64)
    pad = (-t.shape[-1]) % 16
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    return t.reshape(-1, 16)


def check_bfp8_group(actual, exact):
    host = ttnn.to_torch(ttnn.from_torch(exact.to(torch.float32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
    a, h, x = _groups(actual), _groups(host), _groups(exact)
    gmax = x.abs().amax(dim=1, keepdim=True)
    _, exp = torch.frexp(gmax)
    step = torch.where(gmax > 0, torch.pow(2.0, (exp - 7).to(torch.float64)), torch.zeros_like(gmax))
    bad = (a - h).abs() > step
    assert int(bad.sum()) == 0, f"{int(bad.sum())}/{bad.numel()} values beyond one step of their bfp8 group"


def run_case(device, shape, dst, mem, call="positional"):
    x = make_input(shape, dst)
    expected = torch_reference(x, dst)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)

    if call == "positional":
        tt_out = g6_typecast(tt_x, dst)
    elif call == "sub_core_grids":
        tt_out = g6_typecast(tt_x, dst, sub_core_grids=None)
    else:
        tt_out = g6_typecast(tt_x, dtype=dst, sub_core_grids=None)

    # output contract: shape, dtype, layout, memory config of the input
    assert list(tt_out.shape) == list(shape)
    assert tt_out.dtype == dst
    assert tt_out.layout == tt_x.layout
    assert tt_out.memory_config() == tt_x.memory_config()

    actual = ttnn.to_torch(tt_out)
    assert list(actual.shape) == list(shape)
    if dst == ttnn.bfloat8_b:
        check_bfp8_group(actual, expected)
    else:
        check_exact(actual, expected, dst)
    return actual


# (case id, shape, destination, memory config, call form) — production cases only
CASES = [
    ("Q2227_single_tile_l1", (1, 1, 12), ttnn.float32, L1, "positional"),
    ("Q2191_multi_batch", (16, 1, 12), ttnn.float32, DRAM, "positional"),
    ("Q2201_h_and_w_non_aligned", (1, 33, 12), ttnn.float32, DRAM, "positional"),
    ("Q2185_large_multi_batch", (16, 4096, 12), ttnn.float32, DRAM, "positional"),
    ("Q2229_int32", (1, 1, 32, 128), ttnn.int32, DRAM, "sub_core_grids"),
    ("Q2197_bf8b_h_non_aligned", (1, 1, 31, 256), ttnn.bfloat8_b, DRAM, "positional"),
    ("Q2187_bf8b_non_square", (1, 1, 3844, 256), ttnn.bfloat8_b, DRAM, "positional"),
    ("P22_bf8b_l1_multi_batch", (1, 32, 64, 256), ttnn.bfloat8_b, L1, "positional"),
]


@pytest.mark.parametrize("case_id, shape, dst, mem, call", CASES, ids=[c[0] for c in CASES])
def test_g6_typecast(device, case_id, shape, dst, mem, call):
    run_case(device, shape, dst, mem, call)


def test_g6_typecast_keyword_form(device):
    run_case(device, (1, 1, 32, 128), ttnn.int32, DRAM, call="keyword")


def test_g6_typecast_repeat_is_bit_identical(device):
    """BB-06/BB-07: a repeated call (program cache hit) returns the same bits."""
    first = run_case(device, (8, 128, 12), ttnn.float32, DRAM)
    second = run_case(device, (8, 128, 12), ttnn.float32, DRAM)
    assert torch.equal(first.view(torch.int32), second.view(torch.int32))


def test_g6_typecast_validation(device, expect_error):
    x = torch.randn(1, 1, 32, 32).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_typecast(tt_x, "float32")
    with expect_error((ValueError, RuntimeError), ""):
        g6_typecast(
            tt_x,
            ttnn.float32,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        )
    x5 = torch.randn(1, 1, 1, 32, 32).to(torch.bfloat16)
    tt_x5 = ttnn.from_torch(x5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_typecast(tt_x5, ttnn.float32)
