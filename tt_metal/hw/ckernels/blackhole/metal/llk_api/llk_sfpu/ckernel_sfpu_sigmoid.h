// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_improved_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_<true>(-x);
    }

    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator);
    } else {
        result = _sfpu_reciprocal_<1>(denominator);
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

// sigmoid is anti-symmetric and offset by 1
// sigmoid[-x] = 1 - sigmoid[x]
sfpi_inline sfpi::vFloat _sfpu_sigmoid_legacy_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    sfpi::vFloat x = sfpi::abs(val);

    // Polynomial approximation of sigmoid on [0; +inf]
    result = _sigmoid_piecewise_linear_positive_(x);

    // Sigmoid is anti-symmetric and offset by 1.
    // If input was negative then subtract result from 1.0f to get the correct result
    v_if(val < sfpi::vConst0) { result = sfpi::vConst1 - result; }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() {
    if constexpr (!APPROXIMATION_MODE) {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(val);

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
        calculate_sigmoid_appx<ITERATIONS>();
    }
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    if constexpr (!APPROXIMATION_MODE) {
        _init_sfpu_reciprocal_<false>();
    } else {
        sigmoid_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
