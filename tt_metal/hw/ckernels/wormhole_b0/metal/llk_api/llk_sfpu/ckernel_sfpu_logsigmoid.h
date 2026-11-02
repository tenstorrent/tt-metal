// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {
namespace sfpu {

// Mid-range polynomial. P(t) = sum c_k t^k approximates softplus(t) on [-4, 4], and the kernel
// evaluates -P(-x) = sum d_k x^k directly in x. The coefficients are sign-adjusted per
// d_k = -(-1)^k c_k: even k change sign, odd k keep it.
constexpr float LOGSIGMOID_D0 = -0.6924354434013367f;      // -c0
constexpr float LOGSIGMOID_D1 = 0.49275708198547363f;      // +c1
constexpr float LOGSIGMOID_D2 = -0.12142381817102432f;     // -c2
constexpr float LOGSIGMOID_D3 = 0.0031102809589356184f;    // +c3
constexpr float LOGSIGMOID_D4 = 0.00330807245336473f;      // -c4
constexpr float LOGSIGMOID_D5 = -0.00028794066747650504f;  // +c5
constexpr float LOGSIGMOID_D6 = -5.3185409342404455e-05f;  // -c6, held in vConstFloatPrgm0
constexpr float LOGSIGMOID_D7 = 7.1853546614875086e-06f;   // +c7, held in vConstFloatPrgm1
constexpr float LOGSIGMOID_D8 = -7.4961114648886e-08f;     // -c8, held in vConstFloatPrgm2

// logsigmoid(x) = -softplus(-x), with exp(-x) supplied by the caller in dst_index_in1. Needs
// logsigmoid_init first: d6..d8 are read from vConstFloatPrgm0..2, which every exp init also programs.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logsigmoid(
    const std::uint32_t dst_index_in0,  // Index for input (x)
    const std::uint32_t dst_index_in1,  // Index for exp(-x)
    const std::uint32_t dst_index_out)  // Index for output
{
    // d2..d5 stay in the four LRegs the row leaves free: sfpi re-emits a literal's SFPLOADI pair on
    // every row. d0 and d1 stay literals, since one more live value no longer fits in the LRegs.
    const sfpi::vFloat d2 = LOGSIGMOID_D2;
    const sfpi::vFloat d3 = LOGSIGMOID_D3;
    const sfpi::vFloat d4 = LOGSIGMOID_D4;
    const sfpi::vFloat d5 = LOGSIGMOID_D5;

    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Read inputs from destination registers
        sfpi::vFloat x = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat exp_neg_x = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // x <= -4: logsigmoid(x) ~= x (passthrough)
        sfpi::vFloat result = x;

        v_if(x > 4.0f) {
            // For very positive: logsigmoid(x) ~= -exp(-x)
            result = -exp_neg_x;
        }
        v_elseif(x > -4.0f) {
            // Mid-range: sum d_k x^k = -P(-x), bit-identical to negating x and the result
            // since negation commutes with every SFPMAD rounding.
            result = PolynomialEvaluator::eval(
                x,
                LOGSIGMOID_D0,
                LOGSIGMOID_D1,
                d2,
                d3,
                d4,
                d5,
                sfpi::vConstFloatPrgm0,
                sfpi::vConstFloatPrgm1,
                sfpi::vConstFloatPrgm2);
        }
        v_endif;
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void logsigmoid_init() {
    sfpi::vConstFloatPrgm0 = LOGSIGMOID_D6;
    sfpi::vConstFloatPrgm1 = LOGSIGMOID_D7;
    sfpi::vConstFloatPrgm2 = LOGSIGMOID_D8;
}

}  // namespace sfpu
}  // namespace ckernel
