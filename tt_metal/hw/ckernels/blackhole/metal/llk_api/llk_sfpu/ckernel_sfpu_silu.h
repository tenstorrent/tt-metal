// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        // For x <= -87.0f, sigmoid(x) ~ exp(x) and x*exp(x) stays normal down to x = -91.83f
        sfpi::vFloat result;
        v_if (x <= -87.0f) {
            sfpi::vFloat exp_x;
            if constexpr (is_fp32_dest_acc_en) {
                exp_x = _sfpu_exp_accurate_<true>(x);
            } else {
                exp_x = _sfpu_exp_21f_bf16_<true>(x);
            }
            result = x * exp_x;
        } v_else {
            result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
        }
        v_endif;

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must use non-approx sigmoid_init
    sigmoid_init<false>();
}

}  // namespace ckernel::sfpu
