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
        }
        v_endif;
        dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void logsigmoid_init() {}

}  // namespace sfpu
}  // namespace ckernel
