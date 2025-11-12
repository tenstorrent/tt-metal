// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Polynomial approximation for softplus (used in log sigmoid for non-exponential ranges)
inline vFloat softplus_poly_for_logsigmoid(vFloat x) {
    vFloat result = x;  // Default: return x (identity) for x >= 4.0

    v_if(x < 0.0f) {
        // Coefficients for [-4, 0]
        result =
            ((((((-6.11343628e-05 * x - 9.83003622e-04) * x - 4.84124664e-03) * x + 4.19676832e-03) * x +
               1.30285097e-01) *
                  x +
              5.01969907e-01) *
                 x +
             6.93148958e-01);
    }
    v_elseif(x < 4.0f) {
        // Coefficients for [0, 4] - 3rd degree polynomial (Remez-style Minimax)
        result = (((-1.2162164682e-02 * x + 1.3015606879e-01) * x + 5.0493554482e-01) * x + 6.9174850982e-01);
    }
    v_endif;
    // For x >= 4.0: return x (identity - softplus(x) ≈ x for large positive x)

    return result;
}

// LogSigmoid computation: logsigmoid(x) = -softplus(-x) = -(1/beta) * log(1 + exp(-beta * x))
// This kernel uses pre-computed values similar to binary operations
// dst_reg[dst_index_in0] contains: beta * x (scaled input)
// dst_reg[dst_index_in1] contains: exp(-beta * x) (pre-computed)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logsigmoid(
    const uint dst_index_in0,  // Index for scaled input (beta * x)
    const uint dst_index_in1,  // Index for exp(-beta * x)
    const uint dst_index_out,  // Index for output
    uint param0,               // beta (as uint32_t)
    uint param1)               // threshold (as uint32_t)
{
    float beta = Converter::as_float(param0);
    float threshold = Converter::as_float(param1);
    float beta_recip = 1.0f / beta;

    // Process elements one by one
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size_sfpi = 32;

        // Read inputs from destination registers
        vFloat scaled_x = dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        vFloat exp_neg_x = dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Compute neg_scaled_x = -beta * x
        vFloat neg_scaled_x = -scaled_x;

        // Apply logsigmoid logic
        v_if(neg_scaled_x > -threshold) {
            v_if(neg_scaled_x < -4.0f) {
                // For very negative: use exp
                dst_reg[dst_index_out * dst_tile_size_sfpi] = -beta_recip * exp_neg_x;
            }
            v_elseif(neg_scaled_x > 4.0f) {
                // For large positive: use identity
                dst_reg[dst_index_out * dst_tile_size_sfpi] = -beta_recip * neg_scaled_x;
            }
            v_else {
                // For intermediate: use polynomial
                dst_reg[dst_index_out * dst_tile_size_sfpi] = -beta_recip * softplus_poly_for_logsigmoid(neg_scaled_x);
            }
            v_endif;
        }
        v_else {
            // Beyond threshold: return 0
            dst_reg[dst_index_out * dst_tile_size_sfpi] = 0.0f;
        }
        v_endif;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void logsigmoid_init() {
    // Initialize exp with fast+approx mode for computing exp(-x)
    //_init_exponential_<true, true, 0x3F800000>();
}

}  // namespace sfpu
}  // namespace ckernel
