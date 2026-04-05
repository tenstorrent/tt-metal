// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_swish() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Compute exp(-x)
        sfpi::vFloat exp_neg_x = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(-x);

        // Compute denominator: 1 + exp(-x)
        sfpi::vFloat denom = exp_neg_x + sfpi::vConst1;

        // Compute reciprocal of denominator: sigmoid(x) = 1 / (1 + exp(-x))
        sfpi::vFloat sigmoid_x = _sfpu_reciprocal_<2>(denom);

        // Result: x * sigmoid(x)
        sfpi::dst_reg[0] = x * sigmoid_x;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void swish_init() {
    _init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>();
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
