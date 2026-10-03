# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tail coverage for ttnn.erfc.

`test_math.py::test_erfc` draws random bf16 inputs from [0, 1), which is inside the
rational fit's accurate first segment. It therefore cannot see either of the two
things measured here: that the rational's second segment [2.5, 5] is inaccurate,
and that past the fit domain the op used to return a single constant for every
input. Both are asserted against a golden computed from the *unclamped* input.
"""

import torch

import ttnn

# Measured on Blackhole P300, exhaustive over all 65536 bfloat16 patterns:
# 2.007 max BF16 ULP for |x| <= 5 and 2.686 for the asymptotic tail. 4 leaves
# headroom without admitting the 197 ULP the rational produced at x = 5.
_MAX_BF16_ULP = 4.0

# erfc leaves bfloat16 near 9.45, but the last stretch of that is denormal in FP32,
# where the device flushes to zero. 9.2 is the largest input whose true value is
# still an FP32 normal.
_LARGEST_NORMAL_INPUT = 9.2


def _run(values, device, dtype=ttnn.bfloat16):
    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, : values.numel()] = values
    t = ttnn.from_torch(padded, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    return ttnn.to_torch(ttnn.erfc(t))[0, 0, 0, : values.numel()].float()


def _bf16_ulp(got, ref):
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs())) - 7.0)
    return ((got - ref).abs() / ulp).max().item()


def test_erfc_decays_past_the_fit_domain(device):
    """erfc keeps decaying past the rational's domain; it must not go flat."""
    values = torch.arange(2.5, _LARGEST_NORMAL_INPUT + 0.01, 0.25, dtype=torch.float32)
    ref = torch.special.erfc(values.to(torch.bfloat16).double()).float()
    assert (ref > 0).all(), "every reference here must still be a normal number"

    got = _run(values, device)

    # Distinctness first: the failure this guards against is a constant, and a
    # tolerance alone would report it only as a large number.
    assert got.unique().numel() == values.numel(), (
        f"erfc returned {got.unique().numel()} distinct values for {values.numel()} distinct "
        f"inputs in [2.5, {_LARGEST_NORMAL_INPUT}]; it is constant where it should be decaying"
    )
    max_ulp = _bf16_ulp(got.double(), ref.double())
    assert max_ulp <= _MAX_BF16_ULP, f"MaxULP={max_ulp:.2f} exceeds {_MAX_BF16_ULP}"


def test_erfc_underflows_to_zero(device):
    """Past the point erfc leaves the format, the only correct answer is 0."""
    values = torch.tensor([9.5, 10.0, 20.0, 100.0, 3.0e38, float("inf")], dtype=torch.float32)
    got = _run(values, device)
    assert torch.equal(
        got, torch.zeros_like(got)
    ), f"erfc must underflow to 0 once it leaves the format; got {got.tolist()} for {values.tolist()}"


def test_erfc_negative_tail_is_two(device):
    """erfc(-x) = 2 - erfc(x), and for x <= -5 that is exactly 2 in both formats."""
    values = torch.tensor([-5.0, -6.0, -10.0, -20.0, -3.0e38, float("-inf")], dtype=torch.float32)
    got = _run(values, device)
    assert torch.equal(got, torch.full_like(got, 2.0)), f"got {got.tolist()}"


def test_erfc_bfloat8_b_tail_is_a_format_limit_not_a_kernel_one(device):
    """erfc's tail is not representable in bfloat8_b, and that is the format's doing.

    bfloat8_b shares one exponent across a block, so a block that contains erfc(0) = 1
    flushes everything below roughly 2^-8 of it to zero. Measured: the same stimuli that
    give 4.04e-04 ... 4.16e-23 in bfloat16 give all zeros in bfloat8_b. The op is bound
    for bfloat8_b, so this is asserted rather than left as folklore -- it is why the
    decay is checked in bfloat16 only.
    """
    values = torch.arange(2.5, 7.01, 0.5, dtype=torch.float32)
    assert (_run(values, device, dtype=ttnn.bfloat16).unique().numel()) == values.numel()
    assert torch.equal(_run(values, device, dtype=ttnn.bfloat8_b), torch.zeros(values.numel()))
