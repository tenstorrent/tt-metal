// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_threshold.h"

namespace ckernel {
namespace sfpu {

// Op class for threshold: value where x <= threshold, x otherwise. The kernel lives in tt-llk.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, typename T = std::uint32_t>
struct Threshold : SfpuUnaryOp<Threshold<APPROXIMATION_MODE, ITERATIONS, T>> {
    static inline __attribute__((always_inline)) void calculate(const T threshold, const T value) {
        _calculate_threshold_<APPROXIMATION_MODE, ITERATIONS, T>(threshold, value);
    }
};

}  // namespace sfpu
}  // namespace ckernel
