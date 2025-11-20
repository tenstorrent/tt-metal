// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logsigmoid(
    const uint dst_index_in0,  // Index for input (x)
    const uint dst_index_in1,  // Index for exp(-x)
    const uint dst_index_out)  // Index for output
{
    // logsigmoid(x) = -softplus(-x)
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size_sfpi = 32;

        // Read inputs from destination registers
        sfpi::vFloat x = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat exp_neg_x = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Save original x as result; negate x since we compute softplus(-x)
        sfpi::vFloat result = x;
        x = -x;

        v_if(x < -4.0f) {
            // For very negative: use exp
            result = -exp_neg_x;
        }
        v_elseif(x >= -4.0f && x < 4.0f) {
            // Polynomial approximation for softplus(-x) in the mid-range
            result = PolynomialEvaluator::eval(
                x,
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
        v_endif;
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void logsigmoid_init() {}

}  // namespace sfpu
}  // namespace ckernel
