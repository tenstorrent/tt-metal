# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""div where the divisor is a signed zero.

The quotient is in0 * (1 / in1), and the reciprocal of +-0 is +-inf, so the product
is already the IEEE answer except for a subnormal dividend, which the multiply reads
as zero. These pin the dividends the old +0 arm answered wrongly, and the subnormal
dividends a -0 divisor never reached it with.
"""

import math

import pytest
import torch
import ttnn

NAN = float("nan")
INF = float("inf")
SUBNORMAL = 1e-40

DIVIDENDS = [0.0, -0.0, 1.5, -1.5, SUBNORMAL, -SUBNORMAL, INF, -INF, NAN]


def _divide(device, a, b, dtype, torch_dtype):
    ta = torch.full((1, 1, 32, 32), a, dtype=torch_dtype)
    tb = torch.full((1, 1, 32, 32), b, dtype=torch_dtype)
    ia = ttnn.from_torch(ta, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ib = ttnn.from_torch(tb, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.divide(ia, ib)).flatten()[0].item()
    want = (ta / tb).flatten()[0].item()
    return got, want


@pytest.mark.parametrize("b", [0.0, -0.0])
@pytest.mark.parametrize("a", DIVIDENDS)
def test_div_by_zero_fp32(device, a, b):
    got, want = _divide(device, a, b, ttnn.float32, torch.float32)
    if math.isnan(want):
        assert math.isnan(got), f"div({a}, {b}) returned {got}, want nan"
    else:
        same_sign = math.copysign(1.0, got) == math.copysign(1.0, want)
        assert got == want and same_sign, f"div({a}, {b}) returned {got}, want {want}"


# bfloat16 reads a NaN result back as an infinity, so only the dividends whose
# quotient is an infinity are asserted here, and the sign is what is checked.
@pytest.mark.parametrize("b", [0.0, -0.0])
@pytest.mark.parametrize("a", [1.5, -1.5, INF, -INF])
def test_div_by_zero_bf16(device, a, b):
    got, want = _divide(device, a, b, ttnn.bfloat16, torch.bfloat16)
    assert got == want, f"div({a}, {b}) returned {got}, want {want}"
