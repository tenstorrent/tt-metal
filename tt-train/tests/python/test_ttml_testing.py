# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``ttml.testing``: the bf16 ULP metric."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

from ttml.testing import assert_within_ulp, bf16_spacing, ulp_error

BF16_MAX = float((2.0 - 2.0**-7) * 2.0**127)
BF16_MIN_SUBNORMAL = 2.0**-133


def as_bf16(x):
    return np.asarray(x).astype(ml_dtypes.bfloat16)


class TestBf16Spacing:
    @pytest.mark.parametrize(
        "x,expected",
        [(1.0, 2.0**-7), (1.5, 2.0**-7), (2.0, 2.0**-6), (4.0, 2.0**-5), (100.0, 0.5), (0.1, 2.0**-11)],
    )
    def test_known_magnitudes(self, x, expected):
        assert float(bf16_spacing(x)) == expected

    def test_matches_nextafter_across_the_whole_range(self):
        """Independent oracle: the gap ml_dtypes reports above each value."""
        mantissas = np.array([1.0, 1.0 + 2.0**-7, 1.5, 2.0 - 2.0**-7, 2.0 - 2.0**-8, 2.0 - 2.0**-9])
        grid = np.outer(2.0 ** np.arange(-140.0, 127.0), mantissas).ravel()
        rng = np.random.default_rng(0)
        sample = rng.uniform(-1, 1, 20000) * 2.0 ** rng.integers(-140, 126, 20000)
        x = np.abs(np.concatenate([grid, sample]).astype(np.float32)).astype(np.float64)
        x = x[(x > 0) & (x < BF16_MAX)]

        oracle = np.spacing(as_bf16(x)).astype(np.float64)
        mismatch = np.flatnonzero(bf16_spacing(x) != oracle)
        assert mismatch.size == 0, f"{mismatch.size}/{x.size} differ, first at x={x[mismatch[:1]]}"

    @pytest.mark.parametrize("x", [0.0, BF16_MIN_SUBNORMAL, 2.0**-130, 1e-40, 2.0**-126, 2.0**-125 - 2.0**-133])
    def test_floors_at_the_subnormal_spacing(self, x):
        """Below ``2**-125`` the grid stops halving, so every magnitude there shares one gap."""
        assert float(bf16_spacing(x)) == BF16_MIN_SUBNORMAL

    def test_resumes_halving_above_the_subnormal_range(self):
        assert float(bf16_spacing(2.0**-125)) == 2.0**-132

    @pytest.mark.parametrize(
        "x,expected",
        [
            (3.9, 2.0**-6),  # stays in [2, 4)
            (3.998, 2.0**-5),  # rounds up to 4.0, whose gap is twice as wide
            (2.0 - 2.0**-8, 2.0**-6),  # exact midpoint: ties-to-even goes up onto 2.0
            (np.nextafter(2.0 - 2.0**-8, 0.0), 2.0**-7),  # one f64 ulp below the midpoint: stays
        ],
    )
    def test_uses_the_binade_x_rounds_into(self, x, expected):
        assert float(bf16_spacing(x)) == expected

    def test_stays_finite_at_the_top_of_the_range(self):
        """``nextafter`` would report inf here. An inf denominator makes ``ulp_error`` return 0,
        i.e. a vacuous pass, so the grid is extrapolated instead."""
        assert np.isfinite(float(bf16_spacing(BF16_MAX)))
        assert np.isfinite(float(bf16_spacing(1e300)))

    def test_is_sign_symmetric_and_shape_preserving(self):
        assert np.array_equal(bf16_spacing([-7.3, 7.3]), bf16_spacing([7.3, 7.3]))
        assert np.shape(bf16_spacing(1.0)) == ()
        assert bf16_spacing(np.zeros((2, 3))).shape == (2, 3)
        assert float(bf16_spacing(3)) == float(bf16_spacing(np.float32(3)))

    def test_is_monotonic(self):
        assert np.all(np.diff(bf16_spacing(np.logspace(-45, 38, 100000))) >= 0)


class TestUlpError:
    def test_exact_match_scores_zero(self):
        x = np.linspace(-4.0, 4.0, 512)
        assert ulp_error(x, x) == (0.0, 0.0)

    def test_the_two_denominators(self):
        """Hand-checkable: one absolute shift is 1 ULP at the peak but 4 ULP at the smallest
        element, so the peak-normalised and per-element numbers must disagree."""
        expected = np.array([1.0, 2.0, 4.0])
        peak, p99 = ulp_error(expected + 2.0**-5, expected)
        assert peak == 1.0  # 2**-5 / spacing(4.0)
        assert p99 == pytest.approx(3.96, abs=0.01)  # 99th pct of [4, 2, 1]

    def test_the_peak_denominator_ignores_got(self):
        """Scale comes from ``expected`` alone; a wildly wrong output must not be able to inflate
        its own denominator and loosen the bound it is measured against."""
        peak, _ = ulp_error(np.array([1e30]), np.array([1.0]))
        assert peak == pytest.approx((1e30 - 1.0) / 2.0**-7, rel=1e-9)

    def test_survives_an_all_zero_oracle(self):
        zeros = np.zeros((4, 4))
        assert ulp_error(zeros, zeros) == (0.0, 0.0)
        assert np.isfinite(ulp_error(np.full((4, 4), 1e-30), zeros)).all()

    def test_the_peak_scale_is_not_floored(self):
        """A zero oracle still measures against the subnormal spacing. Flooring the scale away
        from zero divides by a spacing 2**26 too wide, which hides the error rather than
        guarding a division ``bf16_spacing`` already cannot make by zero."""
        peak, _ = ulp_error(np.full((4, 4), 1e-30), np.zeros((4, 4)))
        assert peak == pytest.approx(1e-30 / BF16_MIN_SUBNORMAL, rel=1e-9)

    def test_normalises_before_subtracting(self):
        """Lists have no ``-``, and integer subtraction wraps: uint8 0 - 255 measures as 1."""
        assert ulp_error([1.0, 2.0], [1.0, 2.0]) == (0.0, 0.0)
        peak, _ = ulp_error(np.array([0], np.uint8), np.array([255], np.uint8))
        assert peak == 255.0 / float(bf16_spacing(255.0))


class TestAssertWithinUlp:
    def test_reports_both_numbers_and_limits_on_failure(self, expect_error):
        ulp_limit = 0.5  # round-to-nearest bf16; halving every element is far outside it
        truth = np.linspace(1.0, 4.0, 256)
        pattern = rf"lbl: ulp=[\d.]+ \(limit {ulp_limit}\), ulp_p99=[\d.]+ \(limit {ulp_limit}\)"
        with expect_error(AssertionError, pattern):
            assert_within_ulp(truth * 0.5, truth, "lbl", max_ulp=ulp_limit, max_ulp_p99=ulp_limit)

    def test_rejects_a_shape_mismatch_before_measuring(self, expect_error):
        with expect_error(AssertionError, "shape"):
            assert_within_ulp(np.zeros(4), np.zeros(5), "lbl", max_ulp=0.0)  # never reached

    def test_p99_is_opt_in(self):
        """Callers comparing against an oracle that passes near zero omit the p99 limit; flat
        noise must then pass on the peak number alone."""
        ulp_limit = 1.0  # the noise below spans one peak ULP, so the peak number cannot exceed 1
        truth = np.random.default_rng(0).uniform(-4.0, 4.0, (64, 64))
        noisy = truth + np.random.default_rng(1).uniform(-1, 1, truth.shape) * float(bf16_spacing(np.abs(truth).max()))
        assert ulp_error(noisy, truth)[1] > ulp_limit
        assert_within_ulp(noisy, truth, "peak only", max_ulp=ulp_limit)


class TestMetricSensitivity:
    """Pins what the ULP bound catches, and which of the two numbers catches it."""

    @staticmethod
    def truth():
        return np.random.default_rng(0).uniform(-4.0, 4.0, size=(1, 1, 64, 64))

    def test_accepts_bf16_output_quantization(self):
        """Rounding the result to bf16 is the irreducible error of any correct kernel."""
        ulp_limit = 0.5  # round-to-nearest cannot land further than half a ULP from the input
        truth = self.truth()
        rounded = truth.astype(ml_dtypes.bfloat16).astype(np.float64)
        assert_within_ulp(rounded, truth, "bf16 round-trip", max_ulp=ulp_limit, max_ulp_p99=ulp_limit)

    def test_rejects_uniform_rescale(self, expect_error):
        """It is the shape a wrong collective or a missing 1/tp would take."""
        ulp_limit = 0.5  # the bf16 rounding bound a correct kernel meets
        truth = self.truth()
        with expect_error(AssertionError, "ulp="):
            assert_within_ulp(truth * 0.5, truth, "rescaled", max_ulp=ulp_limit, max_ulp_p99=ulp_limit)

    def test_rejects_single_corrupt_element(self, expect_error):
        """One element in 4096 sits inside the p99's discarded tail, so only the peak sees it."""
        ulp_limit = 0.5  # the bf16 rounding bound a correct kernel meets
        truth = self.truth()
        corrupt = truth.copy()
        corrupt[0, 0, 7, 13] += float(np.abs(truth).max())
        assert ulp_error(corrupt, truth)[1] == 0.0
        with expect_error(AssertionError, "ulp="):
            assert_within_ulp(corrupt, truth, "one bad element", max_ulp=ulp_limit, max_ulp_p99=ulp_limit)

    def test_rejects_absolute_noise_at_one_ulp_of_peak(self, expect_error):
        """Kernel error scales with each element's own magnitude; flat noise at the peak's ULP
        is far larger for small elements, and only the p99 guard distinguishes them."""
        ulp_limit = 1.0  # the noise spans one peak ULP, so the peak number cannot exceed 1
        truth = self.truth()
        one_ulp = float(bf16_spacing(np.abs(truth).max()))
        noise = np.random.default_rng(1).uniform(-one_ulp, one_ulp, truth.shape)
        peak, p99 = ulp_error(truth + noise, truth)
        assert peak <= ulp_limit < p99  # the failure text names both numbers, so only this pins which one fired
        with expect_error(AssertionError, "ulp_p99="):
            assert_within_ulp(truth + noise, truth, "flat noise", max_ulp=ulp_limit, max_ulp_p99=ulp_limit)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_rejects_a_non_finite_element(self, bad, expect_error):
        ulp_limit = 0.5  # the bf16 rounding bound a correct kernel meets
        truth = self.truth()
        corrupt = truth.copy()
        corrupt[0, 0, 0, 0] = bad
        assert not ulp_error(corrupt, truth)[0] <= ulp_limit  # `not <=`, since NaN > limit is false too
        with expect_error(AssertionError, "ulp="):
            assert_within_ulp(corrupt, truth, "non-finite", max_ulp=ulp_limit, max_ulp_p99=ulp_limit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
