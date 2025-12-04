// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_unary_max_min.h"

namespace ckernel {
namespace sfpu {

// relu_min(x, threshold) = max(x, threshold)
// Ensures output is at least threshold
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void relu_min(uint uint_threshold) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_float(uint_threshold);
        calculate_unary_max_min_float_body<true>();  // max
        sfpi::dst_reg++;
    }
}

// relu_max(x, threshold) = max(min(x, threshold), 0)
// Clamps to upper bound first, then ensures non-negative
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void relu_max(uint uint_threshold) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_float(uint_threshold);       // Load threshold
        calculate_unary_max_min_float_body<false>();  // x = min(x, threshold)
        load_value_param_float(0);                    // Load 0.0f
        calculate_unary_max_min_float_body<true>();   // x = max(x, 0)
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lrelu(uint slope) {
    _calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS, slope);
}

}  // namespace sfpu
}  // namespace ckernel
