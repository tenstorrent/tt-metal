# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NaN propagation through maximum, minimum, clamp and hardtanh (issue #51470).

All four are built on SFPSWAP, which orders by sign-magnitude: +NaN sorts above +inf and -NaN
below -inf, so a bare swap returns the other operand for maximum(-NaN, x) and minimum(+NaN, x),
and a clamp always returns a bound. The NaN cases run in float32 because a bfloat16 NaN already
comes back as inf through the SFPU store on this path, the same as ttnn.identity.
"""

import math

import pytest
import torch

import ttnn

SHAPE = (1, 1, 32, 64)
NAN_SIGNS = pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive_nan", "negative_nan"])


def nan_input(sign):
    """Every other element is NaN of the given sign; the rest cover finite, zero and infinite values.

    -0.0 is left out: a tie between -0.0 and a bound of 0.0 is a separate question from NaN.
    """
    torch.manual_seed(0)
    x = (torch.rand(SHAPE) * 20 - 10).flatten()
    x[1::2] = math.copysign(float("nan"), sign)
    x[0:8:2] = torch.tensor([0.0, -3.0, float("inf"), -float("inf")])
    return x.reshape(SHAPE)


def to_device(t, device, dtype=ttnn.float32):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def assert_nan_where_input_nan(got, x, want):
    nan = x.isnan()
    assert got[nan].isnan().all(), f"{int((~got[nan].isnan()).sum())} of {int(nan.sum())} NaN inputs came back not NaN"
    assert torch.equal(got[~nan], want[~nan])


@NAN_SIGNS
@pytest.mark.parametrize("op", ["maximum", "minimum"])
@pytest.mark.parametrize("nan_first", [True, False], ids=["nan_lhs", "nan_rhs"])
def test_binary_max_min_nan(device, sign, op, nan_first):
    x = nan_input(sign)
    y = torch.full(SHAPE, 5.0)
    y.view(-1)[0:8:2] = torch.tensor([-5.0, float("inf"), -float("inf"), 0.0])
    a, b = (x, y) if nan_first else (y, x)

    got = ttnn.to_torch(getattr(ttnn, op)(to_device(a, device), to_device(b, device)))

    assert_nan_where_input_nan(got, x, getattr(torch, op)(a, b))


@NAN_SIGNS
@pytest.mark.parametrize("op", ["maximum", "minimum"])
@pytest.mark.parametrize("scalar", [-5.0, 0.0, 5.0])
def test_unary_max_min_nan(device, sign, op, scalar):
    x = nan_input(sign)

    got = ttnn.to_torch(getattr(ttnn, op)(to_device(x, device), scalar))

    assert_nan_where_input_nan(got, x, getattr(torch, op)(x, torch.tensor(scalar)))


@NAN_SIGNS
@pytest.mark.parametrize("bounds", [(-2.0, 2.0), (0.0, 6.0), (0.0, 0.0)])
def test_clamp_hardtanh_nan(device, sign, bounds):
    """torch.clamp and hardtanh pass a NaN input through with its sign, so compare bits."""
    x = nan_input(sign)
    lo, hi = bounds
    tx = to_device(x, device)

    got_clamp = ttnn.to_torch(ttnn.clamp(tx, min=lo, max=hi))
    got_hardtanh = ttnn.to_torch(ttnn.hardtanh(tx, min_val=lo, max_val=hi))

    want = torch.clamp(x, lo, hi)
    assert torch.equal(got_clamp.view(torch.int32), want.view(torch.int32))
    assert torch.equal(got_hardtanh.view(torch.int32), want.view(torch.int32))


@NAN_SIGNS
@pytest.mark.parametrize("bounds", [(0.0, 0.0), (0.0, 5.0), (-1.0, 1.0), (2.0, -2.0)])
def test_clamp_tensor_bounds_nan(device, sign, bounds):
    """The tensor-bounds clamp is minimum then maximum, so it propagates NaN once both do."""
    x = nan_input(sign)
    lo = torch.full(SHAPE, bounds[0])
    hi = torch.full(SHAPE, bounds[1])

    got = ttnn.to_torch(ttnn.clamp(to_device(x, device), to_device(lo, device), to_device(hi, device)))

    assert_nan_where_input_nan(got, x, torch.clamp(x, lo, hi))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_max_min_clamp_finite_unchanged(device, dtype):
    """The guards must not move any non-NaN result: bit-exact against torch on random inputs."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    x = (torch.randn(SHAPE) * 4).to(torch_dtype)
    y = (torch.randn(SHAPE) * 4).to(torch_dtype)
    lo = (torch.rand(SHAPE) * 4 - 2).to(torch_dtype)
    hi = (torch.rand(SHAPE) * 4 + 1).to(torch_dtype)
    tx, ty, tlo, thi = (to_device(t, device, dtype) for t in (x, y, lo, hi))

    cases = [
        (ttnn.maximum(tx, ty), torch.maximum(x, y)),
        (ttnn.minimum(tx, ty), torch.minimum(x, y)),
        (ttnn.maximum(tx, 0.5), torch.maximum(x, torch.tensor(0.5, dtype=torch_dtype))),
        (ttnn.minimum(tx, 0.5), torch.minimum(x, torch.tensor(0.5, dtype=torch_dtype))),
        (ttnn.clamp(tx, min=-1.0, max=1.0), torch.clamp(x, -1.0, 1.0)),
        (ttnn.hardtanh(tx), torch.nn.functional.hardtanh(x)),
        (ttnn.clamp(tx, tlo, thi), torch.clamp(x, lo, hi)),
    ]
    for i, (got, want) in enumerate(cases):
        got = ttnn.to_torch(got)
        assert torch.equal(got, want), f"case {i}: {int((got != want).sum())} of {want.numel()} differ"
