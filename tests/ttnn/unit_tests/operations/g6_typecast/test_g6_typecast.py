# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_typecast (Phase 0: bfloat16 TILE -> float32, bfloat8_b, int32).

Immutable spec for the implementer. Shapes are production cases of
eval/golden_tests/g6_typecast/feature_spec.py (BB-22). Grading follows
helpers.TOLERANCES: every destination but bfloat8_b is bit-exact against
torch.Tensor.to; bfloat8_b is within one quantization step 2^(e_max - 6) of
each group of 16 values against the host's bfloat8_b rendering.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_typecast import g6_typecast

_MEM = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}


def torch_reference(x: torch.Tensor, dtype) -> torch.Tensor:
    if dtype == ttnn.float32:
        return x.to(torch.float32)
    if dtype == ttnn.int32:
        return x.to(torch.int32)  # truncation toward zero
    if dtype == ttnn.bfloat8_b:
        return x.to(torch.float32)  # exact values; rendered in the block format by the check
    raise ValueError(dtype)


def _make_input(shape, dst):
    torch.manual_seed(42)
    if dst == ttnn.int32:
        # fractions and magnitudes up to ~2^30 (|randn| < 8 times 2^27), below 2^31
        x = torch.randn(shape) * torch.pow(2.0, torch.randint(0, 28, shape).to(torch.float32))
        x = x.to(torch.bfloat16)
        flat = x.reshape(-1)
        fixed = [-2.75, -0.5, 0.5, 2.75, 127.5, -1.5, 1.5, -(2.0**31 - 2.0**23), 2.0**30]
        flat[: len(fixed)] = torch.tensor(fixed, dtype=torch.bfloat16)
        return flat.reshape(shape)
    return torch.randn(shape).to(torch.bfloat16)


def _check_contract(out, x_tt, shape, dst):
    assert list(out.shape) == list(shape)
    assert out.dtype == dst
    assert out.layout == x_tt.layout
    assert out.memory_config() == x_tt.memory_config()


def _check_exact(actual, expected):
    if expected.dtype == torch.float32:
        a = actual.to(torch.float32).contiguous().view(torch.int32)
        e = expected.contiguous().view(torch.int32)
    else:
        a = actual.to(torch.int64)
        e = expected.to(torch.int64)
    bad = a != e
    assert not bool(bad.any()), f"{int(bad.sum())}/{bad.numel()} values differ from torch.Tensor.to"


def _groups(t):
    t = t.to(torch.float64)
    pad = (-t.shape[-1]) % 16
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    return t.reshape(-1, 16)


def _check_bfp8_group(actual, exact):
    host = ttnn.to_torch(ttnn.from_torch(exact.to(torch.float32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
    a, h, x = _groups(actual), _groups(host), _groups(exact)
    gmax = x.abs().amax(dim=1, keepdim=True)
    _, exp = torch.frexp(gmax)
    step = torch.where(gmax > 0, torch.pow(2.0, (exp - 7).to(torch.float64)), torch.zeros_like(gmax))
    bad = (a - h).abs() > step
    assert not bool(bad.any()), f"{int(bad.sum())}/{bad.numel()} values beyond one bfloat8_b step of their group"


CASES = [
    # (case id, shape, destination, buffer)
    ("Q2227", (1, 1, 12), ttnn.float32, "l1"),  # single tile, L1
    ("Q2191", (16, 1, 12), ttnn.float32, "dram"),  # multi-batch decode
    ("Q2193", (2, 128, 12), ttnn.float32, "dram"),  # w_non_aligned, multi-tile
    ("Q2201", (1, 33, 12), ttnn.float32, "dram"),  # partial tile row + col
    ("Q2185", (16, 4096, 12), ttnn.float32, "dram"),  # large, flagged p3
    ("Q2197", (1, 1, 31, 256), ttnn.bfloat8_b, "dram"),  # h_non_aligned, non-square
    ("Q2194", (1, 1, 64, 256), ttnn.bfloat8_b, "dram"),  # tile_aligned
    ("Q2212", (1, 1, 129, 256), ttnn.bfloat8_b, "dram"),  # h_non_aligned
    ("P22", (1, 32, 64, 256), ttnn.bfloat8_b, "l1"),  # multi-batch, L1
    ("Q2229", (1, 1, 32, 128), ttnn.int32, "dram"),  # truncation toward zero
]


@pytest.mark.parametrize("case,shape,dst,buffer", CASES, ids=[c[0] for c in CASES])
def test_g6_typecast(device, case, shape, dst, buffer):
    x = _make_input(shape, dst)
    expected = torch_reference(x, dst)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_MEM[buffer])

    if case == "Q2229":
        out = g6_typecast(x_tt, dtype=dst, sub_core_grids=None)  # keyword form
    else:
        out = g6_typecast(x_tt, dst)  # positional form

    _check_contract(out, x_tt, shape, dst)
    actual = ttnn.to_torch(out)
    assert list(actual.shape) == list(shape)
    if dst == ttnn.bfloat8_b:
        _check_bfp8_group(actual, expected)
    else:
        _check_exact(actual, expected)


def test_g6_typecast_repeat_is_bit_identical(device):
    shape = (8, 128, 12)
    x = _make_input(shape, ttnn.float32)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.to_torch(g6_typecast(x_tt, ttnn.float32))
    b = ttnn.to_torch(g6_typecast(x_tt, ttnn.float32))
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))


def test_g6_typecast_rejects_bad_arguments(device, expect_error):
    x = torch.randn((1, 1, 32, 32)).to(torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_typecast(x_tt, "float32")
    with expect_error((ValueError, RuntimeError), ""):
        g6_typecast(
            x_tt,
            ttnn.float32,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        )
