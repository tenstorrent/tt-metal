# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""div_no_nan, which is the divide with one arm replaced.

The contract is that a zero divisor yields zero, including for a zero dividend,
where the plain divide yields NaN. Everything else is the quotient.
"""

import pytest
import torch
import ttnn

ZERO_DIVISOR = [1.0, -1.0, 0.0, -0.0, 7.0, -7.0, 1e30, 1e-30]

# float32 subnormals. They are not zero, so they are not masked: the divide flushes them, as
# test_div_ops.py::test_div_fp32_special_values[div_no_nan] expects.
SUBNORMAL = [1e-40, -1e-40, 2.0**-149, -(2.0**-149)]


def _run(device, dtype, a, b):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    ta = torch.full((1, 1, 32, 32), a, dtype=torch_dtype)
    tb = torch.full((1, 1, 32, 32), b, dtype=torch_dtype)

    ia = ttnn.from_torch(ta, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ib = ttnn.from_torch(tb, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ta, tb, ttnn.to_torch(ttnn.div_no_nan(ia, ib))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("b", [0.0, -0.0])
@pytest.mark.parametrize("a", ZERO_DIVISOR)
def test_div_no_nan_zero_divisor(device, dtype, a, b):
    _, _, got = _run(device, dtype, a, b)

    assert torch.equal(got, torch.zeros_like(got)), f"div_no_nan({a}, {b}) returned {got.flatten()[0]}"
    assert not torch.signbit(got).any(), f"div_no_nan({a}, {b}) returned -0"


@pytest.mark.parametrize("b", SUBNORMAL)
@pytest.mark.parametrize("a", ZERO_DIVISOR + SUBNORMAL)
def test_div_no_nan_fp32_subnormal_divisor(device, a, b):
    ta, tb, got = _run(device, ttnn.float32, a, b)

    tiny = torch.finfo(torch.float32).tiny
    flush = lambda x: torch.where(x.abs() < tiny, torch.copysign(torch.zeros_like(x), x), x)
    expected = flush(ta) / flush(tb)
    expected = torch.where(expected == 0, 0.0, expected)

    assert torch.equal(torch.isnan(got), torch.isnan(expected)), f"div_no_nan({a}, {b}) returned {got.flatten()[0]}"
    finite = ~torch.isnan(expected)
    assert torch.equal(got[finite], expected[finite]), f"div_no_nan({a}, {b}) returned {got.flatten()[0]}"
    assert torch.equal(torch.signbit(got[finite]), torch.signbit(expected[finite]))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_div_no_nan_matches_divide_away_from_zero(device, dtype):
    """Away from a zero divisor it is exactly the divide, which is the point of
    sharing the kernel rather than guarding one with a where."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    a = (torch.rand((1, 1, 32, 32), dtype=torch.float32) * 200 - 100).to(torch_dtype)
    b = (torch.rand((1, 1, 32, 32), dtype=torch.float32) + 0.5).to(torch_dtype)

    ia = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ib = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    assert torch.equal(ttnn.to_torch(ttnn.div_no_nan(ia, ib)), ttnn.to_torch(ttnn.divide(ia, ib)))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_div_no_nan_by_infinity(device, dtype):
    """A zero divisor is the only case the op overrides, so an infinite divisor
    still gives the quotient, which is zero rather than NaN."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    ta = torch.full((1, 1, 32, 32), 2.0, dtype=torch_dtype)
    tb = torch.full((1, 1, 32, 32), float("inf"), dtype=torch_dtype)

    ia = ttnn.from_torch(ta, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ib = ttnn.from_torch(tb, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.div_no_nan(ia, ib)).flatten()[0]

    assert got == 0.0, f"div_no_nan(2, inf) returned {got}"
