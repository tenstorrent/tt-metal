// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "ckernel_sfpu_exp.h"
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
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
    }

    sfpi::vFloat denominator = 1.0f + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = sfpu_reciprocal_iter<2>(denominator);
    } else {
        result = sfpu_reciprocal_iter<1>(denominator);
    }

    return result;
}

// The BF16-grade sigmoid of _sfpu_sigmoid_<false> (exp_21f of -x, then one Newton step of the reciprocal of
// 1 + exp(-x)), with its constants in registers the caller loads once per call, so its loop reloads no FP32 constant.
// (-x) / ln2 is formed as x * -(1 / ln2): the product is the same, sign included. The exponent is clamped at 254
// rather than 255, so exp(-x) stays finite (below 2**128) and the reciprocal needs no NaN guard: from 2**126 up its
// estimate is 0 and the Newton step keeps it 0, the result the guarded step gives for an infinite exp(-x).
struct SigmoidBf16Constants {
    sfpi::vFloat neg_one_ln2 = -1.4426950216293334961f;
    sfpi::vFloat exp_bias = 127.f;
    sfpi::vFloat c0 = EXP_21F_BF16_C0;
    sfpi::vFloat c1 = EXP_21F_BF16_C1;
    sfpi::vFloat c2 = EXP_21F_BF16_C2;
};

template <typename Constants>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_bf16_preloaded_(sfpi::vFloat x, const Constants& k) {
    sfpi::vFloat xlog2 = sfpi::clamp(x * k.neg_one_ln2 + k.exp_bias, 0.0f, 254.0f);
    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
    sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part = sfpi::exman(z);
    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
    frac = PolynomialEvaluator::eval(frac, k.c0, k.c1, k.c2);
    sfpi::vFloat exp_neg_x = sfpi::setexp(frac, exponential_part);
    return sfpu_reciprocal_iter<1, /*normalized=*/true>(1.0f + exp_neg_x);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        const SigmoidBf16Constants constants;
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat result = _sfpu_sigmoid_bf16_preloaded_(sfpi::dst_reg[0], constants);
            // Rounded to BF16 as a vFloat, so the store keeps the destination's format.
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else if constexpr (!APPROXIMATION_MODE) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(val);
            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            }

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
        calculate_sigmoid_appx<ITERATIONS>();
    }
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE) {
        sfpu_reciprocal_init<false>();
    } else {
        sigmoid_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
