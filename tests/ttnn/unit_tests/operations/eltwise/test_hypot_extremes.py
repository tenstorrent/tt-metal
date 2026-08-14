# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ttnn.hypot at operand magnitudes where the naive sqrt(a^2 + b^2)
composition overflows or underflows even though hypot itself is a finite normal value.

The naive composite returns inf for |a| >= 2^64 and 0 for 0 < |a| < 2^-63 in bfloat16
(same magnitude bounds apply to float32; only the representable set differs). torch.hypot
returns a finite value across both bands (see issue #51861).

Also exercises the ±inf edge cases that follow from the rescaled formulation: hypot(inf, inf)
must be inf (the raw m * sqrt(1 + (n/m)²) yields NaN via inf/inf without the override).
"""

import math

import pytest
import torch
import ttnn


def _one_pair(a, b, device, dtype):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    padded_a = torch.zeros((1, 1, 32, 32), dtype=torch_dtype)
    padded_b = torch.zeros((1, 1, 32, 32), dtype=torch_dtype)
    padded_a.view(-1)[0] = float(a)
    padded_b.view(-1)[0] = float(b)
    ta = ttnn.from_torch(padded_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(padded_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ta, tb, padded_a, padded_b


def _rel_ulp(got, ref, mantissa_bits):
    """|got - ref| in units of 1 ULP of |ref| at the target precision."""
    if not math.isfinite(ref) or ref == 0:
        return 0.0 if got == ref else float("inf")
    ulp = 2.0 ** (math.floor(math.log2(abs(ref))) - mantissa_bits)
    return abs(got - ref) / ulp


@pytest.mark.parametrize(
    "a, b",
    [
        (1e25, 0.0),
        (3e38, 0.0),
        (1e-20, 0.0),
        (1e25, 1e25),
        (1e30, 1e30),
        (1e20, 1e20),
        (1e-20, 1e-20),
        (3.0, 4.0),
    ],
)
def test_hypot_extremes_bf16(device, a, b):
    ta, tb, torch_a, torch_b = _one_pair(a, b, device, ttnn.bfloat16)
    got = ttnn.to_torch(ttnn.hypot(ta, tb)).float().view(-1)[0].item()
    ref = torch.hypot(torch_a.float(), torch_b.float()).view(-1)[0].item()
    assert math.isfinite(got), f"hypot({a}, {b}) = {got}, expected finite {ref}"
    if ref == 0.0:
        assert got == 0.0
    else:
        u = _rel_ulp(got, ref, mantissa_bits=7)
        assert u <= 4.0, f"hypot({a}, {b}) = {got}, ref {ref}, {u:.2f} bf16 ULP"


@pytest.mark.parametrize(
    "a, b",
    [
        # Overflow band, |x| > 2^64 ~= 1.84e19
        (1e25, 0.0),
        (3e38, 0.0),
        (2e30, 0.0),
        (1e25, 1e25),
        (1e30, 1e30),
        # Underflow band, |x| < 2^-63 ~= 1.08e-19
        (1e-20, 0.0),
        (1e-30, 0.0),
        (1e-35, 0.0),
        (1e-20, 1e-20),
        (1e-30, 1e-30),
        # Ordinary controls
        (3.0, 4.0),
        (1.0, 0.0),
        (100.0, 100.0),
    ],
)
def test_hypot_extremes_fp32(device, a, b):
    ta, tb, torch_a, torch_b = _one_pair(a, b, device, ttnn.float32)
    got = ttnn.to_torch(ttnn.hypot(ta, tb)).view(-1)[0].item()
    ref = torch.hypot(torch_a.double(), torch_b.double()).view(-1)[0].item()
    # The rewrite adds no rounding beyond an fp32 sqrt/mul/div chain; ~4 fp32 ULP covers
    # the natural error of the formulation.
    assert math.isfinite(got), f"hypot({a}, {b}) = {got}, expected finite {ref}"
    u = _rel_ulp(got, ref, mantissa_bits=23)
    assert u <= 16.0, f"hypot({a}, {b}) = {got}, ref {ref}, {u:.2f} fp32 ULP"


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize(
    "a, b, expected",
    [
        (float("inf"), 0.0, float("inf")),
        (float("inf"), 5.0, float("inf")),
        (0.0, float("inf"), float("inf")),
        (float("inf"), float("inf"), float("inf")),
        (float("-inf"), float("inf"), float("inf")),
        (float("-inf"), float("-inf"), float("inf")),
    ],
)
def test_hypot_inf(device, dtype, a, b, expected):
    """torch.hypot with any ±inf operand returns +inf. The rescaled formula's inf/inf
    resolves to NaN without the override; verify the override handles it."""
    ta, tb, _, _ = _one_pair(a, b, device, dtype)
    got = ttnn.to_torch(ttnn.hypot(ta, tb)).float().view(-1)[0].item()
    assert math.isinf(got) and got > 0, f"hypot({a}, {b}, dtype={dtype}) = {got}, expected +inf"


def test_hypot_zero_zero(device):
    """hypot(0, 0) = 0; the rescaled form must handle max == 0 without NaN."""
    ta, tb, _, _ = _one_pair(0.0, 0.0, device, ttnn.bfloat16)
    got = ttnn.to_torch(ttnn.hypot(ta, tb)).float().view(-1)[0].item()
    assert got == 0.0, f"hypot(0, 0) = {got}, expected 0.0"
