// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// sinh(x) = (exp(x) - exp(-x)) / 2
//
// Implementation strategy:
// We compute exp(x) and exp(-x) independently using the _sfpu_exp_21f_bf16_
// polynomial approximation, then form the difference and halve it.
// This approach naturally preserves the odd symmetry sinh(-x) = -sinh(x)
// through the subtraction, and avoids catastrophic cancellation issues
// because exp(x) and exp(-x) are always on opposite sides of 1.0
// (one >= 1, the other <= 1) for any real x.

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_sinh_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Compute exp(x) and exp(-x) using the exp_21f polynomial approximation
        sfpi::vFloat exp_pos = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(x);
        sfpi::vFloat neg_x = -x;
        sfpi::vFloat exp_neg = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(neg_x);

        // sinh(x) = (exp(x) - exp(-x)) * 0.5
        sfpi::vFloat result = (exp_pos - exp_neg) * 0.5f;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void sinh_init() {
    _init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>();
}

}  // namespace sfpu
}  // namespace ckernel
