// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "xtensor/containers/xarray.hpp"

namespace ttml::test_utils {

// One bf16 ULP at |x|, i.e. the gap between adjacent bf16 values in x's binade, as tests/python/bf16_ulp.py
// computes it: from the exponent, one binade up for mantissas that round into the next power of two, and
// no smaller than at the smallest normal.
inline double bf16_spacing(double x) {
    constexpr int kMantissaBits = 7;
    const double magnitude = std::max(std::abs(x), std::ldexp(1.0, -126));
    int exponent = 0;
    // magnitude = mantissa * 2^exponent, with mantissa in [0.5, 1)
    const double mantissa = std::frexp(magnitude, &exponent);
    const int binade = mantissa >= 1.0 - std::ldexp(1.0, -(kMantissaBits + 2)) ? exponent : exponent - 1;
    return std::ldexp(1.0, binade - kMantissaBits);
}

// |got - expected| in bf16 ULP, as assert_within_bf16_ulp measures it: the peak error in ULP at
// max |expected|, and the p99 of the per-element errors, each in ULP at its own element.
inline void expect_within_bf16_ulp(
    const xt::xarray<float>& got,
    const xt::xarray<double>& expected,
    const std::string& label,
    double max_ulp,
    double max_ulp_p99) {
    ASSERT_EQ(got.shape(), expected.shape()) << label;
    ASSERT_GT(expected.size(), 0U) << label;
    ASSERT_TRUE(xt::all(xt::isfinite(got))) << label << ": non-finite output";

    double peak_error = 0.0;
    double peak_magnitude = 0.0;
    std::vector<double> ulp_per_element(expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::abs(static_cast<double>(got.data()[i]) - expected.data()[i]);
        peak_error = std::max(peak_error, error);
        peak_magnitude = std::max(peak_magnitude, std::abs(expected.data()[i]));
        ulp_per_element[i] = error / bf16_spacing(expected.data()[i]);
    }
    const double ulp = peak_error / bf16_spacing(peak_magnitude);

    // numpy.percentile's default linear interpolation between the two closest ranks.
    const double rank = 0.99 * static_cast<double>(ulp_per_element.size() - 1U);
    const auto lower = static_cast<size_t>(std::floor(rank));
    std::nth_element(ulp_per_element.begin(), ulp_per_element.begin() + lower, ulp_per_element.end());
    const double lower_value = ulp_per_element[lower];
    const double upper_value = lower + 1U < ulp_per_element.size()
                                   ? *std::min_element(ulp_per_element.begin() + lower + 1, ulp_per_element.end())
                                   : lower_value;
    const double ulp_p99 = lower_value + (rank - static_cast<double>(lower)) * (upper_value - lower_value);

    EXPECT_LE(ulp, max_ulp) << label << ": peak error " << ulp << " bf16 ULP";
    EXPECT_LE(ulp_p99, max_ulp_p99) << label << ": p99 error " << ulp_p99 << " bf16 ULP";
}

// The largest |got - expected| over the elements, each in ULP at its own element. A correctly rounded bf16
// result is within 0.5.
inline void expect_each_within_bf16_ulp(
    const xt::xarray<float>& got, const xt::xarray<double>& expected, const std::string& label, double max_ulp) {
    ASSERT_EQ(got.shape(), expected.shape()) << label;
    ASSERT_TRUE(xt::all(xt::isfinite(got))) << label << ": non-finite output";

    double worst_ulp = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double ulp =
            std::abs(static_cast<double>(got.data()[i]) - expected.data()[i]) / bf16_spacing(expected.data()[i]);
        if (ulp > worst_ulp) {
            worst_ulp = ulp;
            worst = i;
        }
    }
    EXPECT_LE(worst_ulp, max_ulp) << label << ": " << worst_ulp << " bf16 ULP at element " << worst << " (got "
                                  << got.data()[worst] << ", expected " << expected.data()[worst] << ")";
}

}  // namespace ttml::test_utils
