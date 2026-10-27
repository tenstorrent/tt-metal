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

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logsigmoid(
    const std::uint32_t dst_index_in0,  // Index for input (x)
    const std::uint32_t dst_index_in1,  // Index for exp(-x)
    const std::uint32_t dst_index_out)  // Index for output
{
    // logsigmoid(x) = -softplus(-x)
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
            // Mid-range: -P(-x), with P the softplus polynomial on [-4, 4]. Evaluated directly
            // in x with coefficients -(-1)^k c_k so neither x nor the result is negated;
            // bit-identical to -P(-x) since negation commutes with every SFPMAD rounding.
            // c6..c8 come from vConstFloatPrgm0..2 (loaded by logsigmoid_init).
            result = PolynomialEvaluator::eval(
                x,
                -0.6924354434013367f,
                0.49275708198547363f,
                -0.12142381817102432f,
                0.0031102809589356184f,
                0.00330807245336473f,
                -0.00028794066747650504f,
                sfpi::vConstFloatPrgm2,
                sfpi::vConstFloatPrgm0,
                sfpi::vConstFloatPrgm1);
        }
        v_endif;
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void logsigmoid_init() {
    // Three highest-order polynomial coefficients (sign-flipped, see calculate_logsigmoid).
    sfpi::vConstFloatPrgm0 = 7.1853546614875086e-06f;   // c7
    sfpi::vConstFloatPrgm1 = -7.4961114648886e-08f;     // c8
    sfpi::vConstFloatPrgm2 = -5.3185409342404455e-05f;  // c6
}

}  // namespace sfpu
}  // namespace ckernel
