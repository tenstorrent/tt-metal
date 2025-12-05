// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// relu_min(x, threshold) = max(x, threshold)
// Ensures output is at least threshold
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void relu_min(uint uint_threshold) {
    // Load threshold outside the loop for better performance
    TT_SFPLOADI(p_sfpu::LREG2, 10, uint_threshold & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, uint_threshold >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // x = max(x, threshold) using LREG2
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);  // store max

        sfpi::dst_reg++;
    }
}

// relu_max(x, threshold) = max(min(x, threshold), 0)
// Clamps to upper bound first, then ensures non-negative
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void relu_max(uint uint_threshold) {
    // Load both params outside the loop for better performance
    // threshold -> LREG2, 0.0f -> LREG3
    TT_SFPLOADI(p_sfpu::LREG2, 10, uint_threshold & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, uint_threshold >> 16);
    TT_SFPLOADI(p_sfpu::LREG3, 10, 0);
    TT_SFPLOADI(p_sfpu::LREG3, 8, 0);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // x = min(x, threshold) using LREG2
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);  // store min

        // x = max(x, 0) using LREG3
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
        TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);  // store max

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lrelu(const uint slope) {
    _calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS, slope);
}

}  // namespace sfpu
}  // namespace ckernel
