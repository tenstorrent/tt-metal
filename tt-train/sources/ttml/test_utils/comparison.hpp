// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <xtensor-blas/xlinalg.hpp>

namespace ttml::test_utils {

// gtest assertion that two xtensor arrays match within tolerances. Pass rtol = atol = 0 for ops whose
// output is bit-exact (e.g. pure data-movement reorders). This is a drop-in for
// `EXPECT_TRUE(xt::allclose(...))`: like xt::allclose it broadcasts, and adds a labeled message.
inline void expect_allclose(
    const xt::xarray<float>& actual,
    const xt::xarray<float>& expected,
    double rtol = 1e-5,
    double atol = 1e-8,
    const std::string& tag = "") {
    EXPECT_TRUE(xt::allclose(actual, expected, rtol, atol)) << tag << ": value mismatch";
}

// gtest assertion on the scale-invariant relative L2 error ||actual - expected|| / ||expected||.
// Preferred over elementwise allclose for matmul-heavy ops, where BF16 noise makes per-element
// tolerances unstable across tensor sizes (see e.g. SwiGLU).
inline void expect_relative_l2(
    const xt::xarray<float>& actual, const xt::xarray<float>& expected, double threshold, const std::string& tag = "") {
    ASSERT_EQ(actual.shape(), expected.shape()) << tag << ": shape mismatch";
    const xt::xarray<float> diff = actual - expected;
    const float diff_l2 = std::sqrt(xt::sum(xt::square(diff))());
    const float ref_l2 = std::sqrt(xt::sum(xt::square(expected))());
    const float rel_l2 = diff_l2 / (ref_l2 + 1e-12F);
    EXPECT_LT(rel_l2, threshold) << tag << ": relative L2 error " << rel_l2 << " >= " << threshold;
}

}  // namespace ttml::test_utils
