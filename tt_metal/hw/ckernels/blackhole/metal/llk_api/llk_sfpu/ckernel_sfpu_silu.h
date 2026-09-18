// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
//
// Evaluating sigmoid first throws the deep negative tail away: exp(-x) reaches 2^126 at
// x = ln(2^-126) = -87.3365 (and overflows to +inf below x = -88.72), so the reciprocal
// flushes to zero and x * 0 is exactly 0 -- although x * exp(x) is a normal float all the
// way down to x = -91.83, and silu is -1.02e-36 at the threshold itself.
//
// Below the threshold the exponential is evaluated at -x - 44 instead, i.e. exp(-x) * e^-44,
// and the numerator is scaled by the same e^-44, which keeps both operands of the reciprocal
// inside the normal range. 44 is chosen because -x lies in [64, 128) over the whole affected
// band, where subtracting an exact integer is itself exact, so the only new error is the one
// rounding of the e^-44 constant (5e-9 relative). The 1 in the denominator is then a
// 1.5e-19 relative perturbation of a quantity >= 6.6e18 and disappears in the fp32 rounding,
// which is exactly the 1/(1+exp(-x)) -> exp(x) limit the tail is in.
constexpr float SILU_TAIL_THRESHOLD = -87.3365f;  // ln(2^-126), where 1 + exp(-x) hits 2^126
constexpr float SILU_TAIL_SHIFT = 44.0f;
constexpr float SILU_TAIL_SCALE = 7.78113228e-20f;  // e^-44

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat arg = -x;
        sfpi::vFloat numerator = x;
        v_if(x < SILU_TAIL_THRESHOLD) {
            arg = -x - SILU_TAIL_SHIFT;
            numerator = x * SILU_TAIL_SCALE;
        }
        v_endif;

        // sigmoid's body, with the argument and numerator above rather than -x and x
        sfpi::vFloat exp_neg_x;
        if constexpr (is_fp32_dest_acc_en) {
            exp_neg_x = _sfpu_exp_accurate_<true>(arg);
        } else {
            exp_neg_x = _sfpu_exp_21f_bf16_<true>(arg);
        }
        sfpi::vFloat denominator = 1.0f + exp_neg_x;

        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en) {
            result = numerator * sfpu_reciprocal_iter<2>(denominator);
        } else {
            result = numerator * sfpu_reciprocal_iter<1>(denominator);
        }

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
    // calculate_silu inlines the non-approx sigmoid path, so we must use non-approx sigmoid_init
    sigmoid_init<false>();
}

}  // namespace ckernel::sfpu
