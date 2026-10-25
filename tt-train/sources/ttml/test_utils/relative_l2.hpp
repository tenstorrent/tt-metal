// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "xtensor/containers/xarray.hpp"
#include "xtensor/core/xmath.hpp"

namespace ttml::test_utils {

// ||a - b|| / ||b||: scale-invariant, so one bound holds across shapes where elementwise tolerances fail on the
// outliers of large tensors or on tiny reference values. 1e-2 is the usual bound for bf16 against an fp32 reference.
inline float relative_l2(const xt::xarray<float>& a, const xt::xarray<float>& b) {
    const float diff_l2 = std::sqrt(xt::sum(xt::square(a - b))());
    const float ref_l2 = std::sqrt(xt::sum(xt::square(b))());
    return diff_l2 / (ref_l2 + 1e-12F);
}

}  // namespace ttml::test_utils
