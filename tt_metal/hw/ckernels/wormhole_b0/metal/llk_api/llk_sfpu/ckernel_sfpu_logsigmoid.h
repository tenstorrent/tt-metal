// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logsigmoid(
    const std::uint32_t dst_index_in0,  // Index for input (x)
    const std::uint32_t dst_index_in1,  // Index for exp(-|x|) / exp(-x)
    const std::uint32_t dst_index_out)  // Index for output
{
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Read inputs from destination registers
        sfpi::vFloat x = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat exp_neg_x = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result;
        // Negate x to evaluate softplus(-x)
        sfpi::vFloat neg_x = -x;

        v_if(neg_x < -4.0f) {
            // For x > 4.0f: log_sigmoid(x) = log(1 / (1 + exp(-x))) ≈ -exp(-x)
            result = -exp_neg_x;
        }
        v_elseif(neg_x <= 4.0f) {
            // For x in [-4.0f, 4.0f]: polynomial approximation of -softplus(-x)
            result = PolynomialEvaluator::eval(
                neg_x,
                0.6924354434013367f,
                0.49275708198547363f,
                0.12142381817102432f,
                0.0031102809589356184f,
                -0.00330807245336473f,
                -0.00028794066747650504f,
                5.3185409342404455e-05f,
                7.1853546614875086e-06f,
                7.4961114648886e-08f);
            result = -result;
        }
        v_else {
            // For x < -4.0f (neg_x > 4.0f): log_sigmoid(x) = x - log1p(exp(x)) ≈ x - exp(x)
            result = x - exp_neg_x;
        }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void logsigmoid_init() {}

}  // namespace sfpu
}  // namespace ckernel
