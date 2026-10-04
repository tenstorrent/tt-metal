# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""pow with a negative base and a large integer exponent.

IEEE 754 defines `pow(x, y)` for `x < 0` whenever `y` is a finite integer: the magnitude is
`|x|**y` and the sign is negative exactly when `y` is odd. The cases below assert that
contract at exponents the existing suite does not reach, where no tolerance is needed:
`(-1)**y` is exactly `-1` or `+1`, and `(-2)**y` overflows to an infinity whose sign is still
set by the parity of `y`.

Non-integer exponents with a negative base stay NaN, and positive bases are untouched; both
are included so that a fix cannot pass by widening the integer test too far.

These cases use a tensor exponent, which is the path through `_sfpu_binary_power_*`. A scalar
exponent is classified separately and is not covered here.
"""

import pytest
import torch
import ttnn

# (exponent, is_odd). Every value is exactly representable in float32:
# the spacing is 1/2 at 2**22, 1 at 2**23 and 2 at 2**24.
_INTEGER_EXPONENTS = [
    (32766.0, False),  # inside a sign-plus-15-bit range
    (32767.0, True),  # the largest value such a range holds
    (32768.0, False),
    (65536.0, False),
    (-40000.0, False),
    (4194305.0, True),  # 2**22 + 1
    (4194306.0, False),  # 2**22 + 2
    (8388609.0, True),  # 2**23 + 1
    (16777216.0, False),  # 2**24
]

_NON_INTEGER_EXPONENTS = [32768.5, 4194304.5]


def _tile(value):
    return torch.full((1, 1, 32, 32), value, dtype=torch.float32)


def _to_device(device, value):
    return ttnn.from_torch(_tile(value), layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=device)


def _run(device, base, exponent):
    base_tt = _to_device(device, base)
    result = ttnn.pow(base_tt, _to_device(device, exponent))
    return ttnn.to_torch(result).double().reshape(-1)


@pytest.mark.parametrize("exponent, is_odd", _INTEGER_EXPONENTS)
def test_pow_negative_one_follows_the_exponent_parity(device, exponent, is_odd):
    """(-1)**y is exactly -1 for odd y and exactly +1 for even y."""
    expected = -1.0 if is_odd else 1.0
    result = _run(device, -1.0, exponent)
    assert (result == expected).all(), (
        f"pow(-1, {exponent:g}): expected {expected:g} for an "
        f"{'odd' if is_odd else 'even'} exponent, got {result[0].item()}"
    )


# Bases whose magnitude is not 1, so the result saturates. Which way it saturates depends on
# the base's magnitude and the exponent's sign, while the parity still sets the sign bit:
# |base| > 1 with y > 0 overflows, with y < 0 underflows, and the other way round for |base| < 1.
_SATURATING_CASES = [
    (-2.0, 32768.0),
    (-2.0, 65536.0),
    (-2.0, 4194305.0),
    (-2.0, 4194306.0),
    (-2.0, 8388609.0),
    (-2.0, -40000.0),
    (-0.5, 40000.0),
    (-0.5, -40000.0),
]


@pytest.mark.parametrize("base, exponent", _SATURATING_CASES)
def test_pow_saturated_result_keeps_the_exponent_parity(device, base, exponent):
    """A saturated magnitude still carries the sign the exponent's parity gives it.

    Compared against torch exactly, including the sign of a zero, because every expected value
    here is representable: an infinity or a zero of one sign or the other.
    """
    expected = torch.pow(torch.tensor(base, dtype=torch.float32), torch.tensor(exponent, dtype=torch.float32))
    result = _run(device, base, exponent)
    assert (result == expected.double()).all() and (torch.signbit(result) == torch.signbit(expected)).all(), (
        f"pow({base:g}, {exponent:g}): expected {expected.item()} "
        f"(signbit {int(torch.signbit(expected))}), got {result[0].item()} "
        f"(signbit {int(torch.signbit(result[0]))})"
    )


@pytest.mark.parametrize("exponent", _NON_INTEGER_EXPONENTS)
def test_pow_negative_base_non_integer_exponent_stays_nan(device, exponent):
    """A negative base with a non-integer exponent has no real result and must stay NaN."""
    result = _run(device, -1.0, exponent)
    assert result.isnan().all(), f"pow(-1, {exponent:g}): expected nan, got {result[0].item()}"


@pytest.mark.parametrize("exponent", [32768.0, 4194306.0, 32768.5])
def test_pow_positive_base_is_unaffected(device, exponent):
    """1**y is 1 for every y; the sign logic must not reach a positive base."""
    result = _run(device, 1.0, exponent)
    assert (result == 1.0).all(), f"pow(1, {exponent:g}): expected 1, got {result[0].item()}"
