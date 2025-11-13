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

// // Polynomial approximation for softplus (used in log sigmoid for non-exponential ranges)
// inline vFloat softplus_poly_for_logsigmoid(vFloat x) {
//     vFloat result = x;  // Default: return x (identity) for x >= 4.0

//     v_if(x < 0.0f) {
//         // Coefficients for [-4, 0]
//         result =
//             ((((((-6.11343628e-05 * x - 9.83003622e-04) * x - 4.84124664e-03) * x + 4.19676832e-03) * x +
//                1.30285097e-01) *
//                   x +
//               5.01969907e-01) *
//                  x +
//              6.93148958e-01);
//     }
//     v_elseif(x < 4.0f) {
//         // Coefficients for [0, 4] - 3rd degree polynomial (Remez-style Minimax)
//         result = (((-1.2162164682e-02 * x + 1.3015606879e-01) * x + 5.0493554482e-01) * x + 6.9174850982e-01);
//     }
//     v_endif;
//     // For x >= 4.0: return x (identity - softplus(x) ≈ x for large positive x)

//     return result;
// }

// LogSigmoid computation: logsigmoid(x) = -softplus(-x) = -(1/beta) * log(1 + exp(-beta * x))
// This kernel uses pre-computed values similar to binary operations
// dst_reg[dst_index_in0] contains: beta * x (scaled input)
// dst_reg[dst_index_in1] contains: exp(-beta * x) (pre-computed)

// Beta = 1 , beta recip = 1
// threshold = 20

// For logsigmoid
// result = -softplus(-x)
// input1 = -x, input2 = exp(-x)
// x = -x
// result = -x
// (x < -4) exp

// (x < 4  )
// {
// poly8
// }

// x > 4 is identity

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logsigmoid(
    const uint dst_index_in0,  // Index for scaled input (beta * x)
    const uint dst_index_in1,  // Index for exp(-beta * x)
    const uint dst_index_out)  // Index for output
// uint param0,               // beta (as uint32_t)
// uint param1)               // threshold (as uint32_t)
{
    float beta = 1.0f;
    float threshold = 4.0f;
    float beta_recip = 1.0f;

    // Process elements one by one
    // logsigmoid(x) = -softplus(-x)
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size_sfpi = 32;

        // Read inputs from destination registers
        vFloat neg_x = dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        vFloat exp_neg_x = dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        vFloat result = -neg_x;

        v_if(neg_x < -4.0f) {
            // For very negative: use exp
            result = -exp_neg_x;
        }
        v_elseif(neg_x >= -4.0f && neg_x < 4.0f) {
            result =
                -(0.6924354434013367f +
                  neg_x * (0.49275708198547363f +
                           neg_x * (0.12142381817102432f +
                                    neg_x * (0.0031102809589356184f +
                                             neg_x * (-0.00330807245336473f +
                                                      neg_x * (-0.00028794066747650504f +
                                                               neg_x * (5.3185409342404455e-05f +
                                                                        neg_x * (7.1853546614875086e-06f +
                                                                                 neg_x * (7.4961114648886e-08f)))))))));
            // result = 2.5f;
        }
        // v_elseif(neg_x >= -4.0f && neg_x < 0.0f) {
        //     // Coefficients for [-4, 0]
        //     result = ((((((-6.11343628e-05 * neg_x - 9.83003622e-04) * neg_x - 4.84124664e-03) * neg_x
        //     + 4.19676832e-03) * neg_x + 1.30285097e-01) * neg_x + 5.01969907e-01) * neg_x + 6.93148958e-01);
        // }
        // v_elseif(neg_x >= 0.0f && neg_x < 4.0f) {
        //     // Coefficients for [0, 4]
        //     result = (((-1.2162164682e-02 * neg_x + 1.3015606879e-01) * neg_x + 5.0493554482e-01) * neg_x
        //     + 6.9174850982e-01);
        // }
        v_endif;

        dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
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
