# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""bf16 ULP error metrics for tt-train tests."""

from __future__ import annotations

import numpy as np

BF16_MANTISSA_BITS = 7
BF16_MIN_NORMAL = 2.0**-126  # bf16 goes subnormal below this and the spacing stops shrinking
# frexp mantissas from here up round to the next power of two, so the spacing is that binade's.
BF16_ROUNDS_UP_FROM = 1.0 - 2.0 ** -(BF16_MANTISSA_BITS + 2)


def bf16_spacing(x):
    """Gap between adjacent bf16 values where ``x`` lands, i.e. one bf16 ULP at that magnitude.

    Computed from the exponent instead of ``np.spacing`` on a bf16 cast, which ml_dtypes gets
    wrong in two ways: it reports inf for anything that rounds to ``BF16_MAX`` or above, and it
    rounds f64 through f32, so a value just below a binade's rounding midpoint is lifted onto
    the midpoint and then ties-to-even into the next binade.
    """
    x = np.maximum(np.abs(np.asarray(x, np.float64)), BF16_MIN_NORMAL)
    mantissa, exponent = np.frexp(x)
    binade = np.where(mantissa >= BF16_ROUNDS_UP_FROM, exponent, exponent - 1)
    return 2.0 ** (binade - BF16_MANTISSA_BITS)


def bf16_ulp_error(got, expected) -> tuple[float, float]:
    """``|got - expected|`` in bf16 ULP.

    Returns ``(peak_ulp, p99_ulp)``: the max error in ULP at ``max |expected|``, then the p99 of
    the per-element errors, each in ULP at its own element.
    """
    got, expected = np.asarray(got, np.float64), np.asarray(expected, np.float64)
    if got.shape != expected.shape:
        raise AssertionError(f"shape {got.shape} != {expected.shape}")
    if expected.size == 0:
        raise AssertionError("expected is empty")
    if not np.isfinite(expected).all():
        raise AssertionError("expected has non-finite elements")
    spacing = bf16_spacing(expected)
    err = np.abs(got - expected)
    return float(err.max() / spacing.max()), float(np.percentile(err / spacing, 99))


def assert_within_bf16_ulp(got, expected, label: str, max_ulp: float, max_ulp_p99: float = np.inf) -> None:
    """Assert ``got`` matches ``expected`` to ``max_ulp`` bf16 ULP at the peak and, if given, to
    ``max_ulp_p99`` at the 99th percentile of the per-element errors.

    Per-element errors are measured in ULP at each ``expected`` value's own magnitude, so where
    the oracle is zero or tiny any residual counts as a huge error and can carry the p99 with it.
    Oracles with such elements (masks, padding, activation tails) need a looser ``max_ulp_p99``,
    or none.
    """
    try:
        ulp, ulp_p99 = bf16_ulp_error(got, expected)
    except AssertionError as e:
        raise AssertionError(f"{label}: {e}") from None
    detail = f"{label}: ulp={ulp:.2f} (limit {max_ulp}), ulp_p99={ulp_p99:.2f} (limit {max_ulp_p99})"
    if not (ulp <= max_ulp and ulp_p99 <= max_ulp_p99):
        raise AssertionError(detail)
