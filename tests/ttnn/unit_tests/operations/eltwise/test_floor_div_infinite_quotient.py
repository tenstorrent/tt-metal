# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""floor_div where the quotient is infinite.

floor_div used to compute the quotient twice, once unrounded, so it could return
the unrounded one when it was infinite. That guard selected the same value the
floored path already produced, because floor leaves an infinity alone. These
tests pin that, so removing the guard cannot become wrong silently: if the floor
step ever stops passing an infinity through, they fail here rather than in a
caller.
"""

import pytest
import torch
import ttnn

# a / b overflows to an infinity while both operands are ordinary numbers, which
# is the only situation in which the removed guard could have fired.
INFINITE_QUOTIENT = [
    (1e30, 1e-30),
    (-1e30, 1e-30),
    (1e30, -1e-30),
    (-1e30, -1e-30),
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("a, b", INFINITE_QUOTIENT)
def test_floor_div_infinite_quotient(device, dtype, a, b):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    ta = torch.full((1, 1, 32, 32), a, dtype=torch_dtype)
    tb = torch.full((1, 1, 32, 32), b, dtype=torch_dtype)
    want = torch.floor_divide(ta, tb).flatten()[0]

    ia = ttnn.from_torch(ta, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ib = ttnn.from_torch(tb, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.floor_div(ia, ib)).flatten()[0]

    assert got == want, f"floor_div({a}, {b}) returned {got}, want {want}"


def test_floor_div_int32(device):
    """int32 floor_div, which the guarded form did not survive.

    The guard divided unrounded first, and that quotient is a float, so the
    int32 path came back as denormals and NaNs. Rounding toward negative
    infinity is the whole point of the op, so the negative cases are the ones
    that matter: -7 // 2 is -4, not -3.
    """
    a = torch.tensor([7, -7, 6, -6, 100, -100, 1, 0] * 128, dtype=torch.int32).reshape(1, 1, 32, 32)
    b = torch.tensor([2, 2, -3, -3, 7, 7, 3, 5] * 128, dtype=torch.int32).reshape(1, 1, 32, 32)
    want = torch.floor_divide(a, b)

    ia = ttnn.from_torch(a, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    ib = ttnn.from_torch(b, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.floor_div(ia, ib))

    assert torch.equal(got, want), f"{int((got != want).sum())} of {want.numel()} differ"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("a, b", INFINITE_QUOTIENT)
def test_floor_div_matches_the_floored_divide(device, dtype, a, b):
    """The guard's premise: that these two can disagree. They do not."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    ta = torch.full((1, 1, 32, 32), a, dtype=torch_dtype)
    tb = torch.full((1, 1, 32, 32), b, dtype=torch_dtype)

    ia = ttnn.from_torch(ta, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ib = ttnn.from_torch(tb, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    guarded = ttnn.to_torch(ttnn.floor_div(ia, ib))
    floored = ttnn.to_torch(ttnn.div(ia, ib, fast_and_approximate_mode=False, rounding_mode="floor"))

    assert torch.equal(guarded, floored), f"floor_div and div(rounding_mode='floor') disagree at ({a}, {b})"
