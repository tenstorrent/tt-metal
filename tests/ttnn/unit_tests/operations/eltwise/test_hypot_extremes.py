# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.hypot at the magnitudes where a^2 leaves the format.

hypot(a, b) is a finite normal for every pair a plain sqrt(a^2 + b^2) cannot
reach: the square overflows for |x| >= 2^64 and stops being normal for
0 < |x| < 2^-63, while hypot itself is still representable. These tests pin the
two bands, the special values, and an exhaustive bfloat16 sweep.
"""

import math

import pytest
import torch

import ttnn

# Everything a bfloat16 can hold, as one 64-tile input.
ALL_BF16 = torch.arange(0, 65536, dtype=torch.int64).to(torch.int32).to(torch.int16).view(torch.bfloat16)

TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def _hypot(device, a, b, dtype):
    ta = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(ttnn.hypot(ta, tb))


def _pair(a, b, dtype):
    t = TORCH_DTYPE[dtype]
    ta = torch.full((1, 1, 32, 32), float(a), dtype=t)
    tb = torch.full((1, 1, 32, 32), float(b), dtype=t)
    return ta, tb


def _ulps(got, ref, mantissa_bits):
    """|got - ref| in units of one ULP of ref."""
    if not math.isfinite(ref) or ref == 0.0:
        return 0.0 if got == ref else float("inf")
    return abs(got - ref) / 2.0 ** (math.floor(math.log2(abs(ref))) - mantissa_bits)


@pytest.mark.parametrize("other", [0.0, 1.0, 1e-30, 1e30, -3.0])
@pytest.mark.parametrize("swap", [False, True])
def test_hypot_bf16_exhaustive(device, other, swap):
    """Every bfloat16 bit pattern against one fixed operand, both orders."""
    swept = ALL_BF16.reshape(1, 1, 256, 256)
    fixed = torch.full((1, 1, 256, 256), other, dtype=torch.bfloat16)
    a, b = (fixed, swept) if swap else (swept, fixed)

    got = _hypot(device, a, b, ttnn.bfloat16).flatten()
    ref = torch.hypot(a.to(torch.float64), b.to(torch.float64)).to(torch.bfloat16).flatten()

    # A bfloat16 DST cannot carry a NaN, and subnormal operands are flushed
    # before the kernel sees them; both are true of the composite this replaces.
    interesting = ~ALL_BF16.isnan() & ~((ALL_BF16 != 0) & (ALL_BF16.abs().to(torch.float32) < 2**-126))

    # The square root is a polynomial, not a correctly rounded operation, so the
    # last bit is allowed to move; nothing beyond it is.
    g, r = got.to(torch.float64), ref.to(torch.float64)
    within = (g == r) | ((r != 0) & torch.isfinite(r) & ((g - r).abs() <= r.abs() * 2**-7))
    wrong = ~within & interesting
    assert not wrong.any(), (
        f"{int(wrong.sum())} of {int(interesting.sum())} patterns off by more than one ULP, "
        f"first at 0x{int(ALL_BF16.view(torch.int16)[wrong.nonzero()[0, 0]].item()) & 0xffff:04x}: "
        f"got {got[wrong][0].item()}, expected {ref[wrong][0].item()}"
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "a, b",
    [
        # a^2 overflows, hypot does not
        (1e25, 0.0),
        (3e38, 0.0),
        (1e25, 1e25),
        (1e30, 1e30),
        (1.9e19, 1.0),
        # a^2 stops being normal, hypot does not
        (1e-20, 0.0),
        (1e-30, 0.0),
        (1e-35, 1e-35),
        (1e-20, 1e-20),
        (1.0e-19, 1.0e-19),
        # in band
        (3.0, 4.0),
        (1.0, 0.0),
        (100.0, 100.0),
    ],
)
def test_hypot_bands(device, dtype, a, b):
    ta, tb = _pair(a, b, dtype)
    got = _hypot(device, ta, tb, dtype).float().view(-1)[0].item()
    ref = torch.hypot(ta.double(), tb.double()).view(-1)[0].item()
    bits = 7 if dtype == ttnn.bfloat16 else 23
    ref = float(torch.tensor(ref, dtype=TORCH_DTYPE[dtype]))
    assert math.isfinite(got), f"hypot({a}, {b}) = {got}, expected {ref}"
    u = _ulps(got, ref, bits)
    assert u <= 1.0, f"hypot({a}, {b}) = {got}, expected {ref}, off by {u:.2f} ULP"


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (float("inf"), 0.0, float("inf")),
        (float("inf"), 5.0, float("inf")),
        (0.0, float("inf"), float("inf")),
        (float("inf"), float("inf"), float("inf")),
        (float("-inf"), float("inf"), float("inf")),
        (float("-inf"), float("-inf"), float("inf")),
        # IEEE 754: an inf operand wins over a NaN one.
        (float("inf"), float("nan"), float("inf")),
        (float("nan"), float("inf"), float("inf")),
        (float("nan"), 1.0, float("nan")),
        (float("nan"), float("nan"), float("nan")),
        (0.0, 0.0, 0.0),
    ],
)
def test_hypot_specials_fp32(device, a, b, expected):
    ta, tb = _pair(a, b, ttnn.float32)
    got = _hypot(device, ta, tb, ttnn.float32).view(-1)[0].item()
    if math.isnan(expected):
        assert math.isnan(got), f"hypot({a}, {b}) = {got}, expected NaN"
    else:
        assert got == expected, f"hypot({a}, {b}) = {got}, expected {expected}"
