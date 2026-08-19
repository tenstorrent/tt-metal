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

template <bool is_fp32_acc_to_dest_mode = true, bool EXP_COEFFS_IN_PRGM_REGS = false>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true, EXP_COEFFS_IN_PRGM_REGS>(-x);
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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() {
    if constexpr (!APPROXIMATION_MODE) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en, /*EXP_COEFFS_IN_PRGM_REGS*/ true>(val);
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
        // Preload the exp_21f polynomial tail coefficients for the
        // EXP_COEFFS_IN_PRGM_REGS fast path in calculate_sigmoid/calculate_silu
        // (Prgm0 is owned by sfpu_reciprocal's 2.0f).
        sfpi::vConstFloatPrgm1 = 7.839635491371155e-08f;
        sfpi::vConstFloatPrgm2 = 4.791750143340323e-15f;
    } else {
        sigmoid_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
