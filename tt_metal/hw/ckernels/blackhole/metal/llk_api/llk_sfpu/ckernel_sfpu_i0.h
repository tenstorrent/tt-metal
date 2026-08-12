// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_polyval.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

#define POLYVAL10(coef10, coef9, coef8, coef7, coef6, coef5, coef4, coef3, coef2, coef1, coef0, t4)               \
    ((coef0 +                                                                                                     \
      (coef1 +                                                                                                    \
       (coef2 +                                                                                                   \
        (coef3 +                                                                                                  \
         (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) * t4) * t4) * \
            t4) *                                                                                                 \
           t4) *                                                                                                  \
          t4) *                                                                                                   \
     t4)

// Asymptotic path for i0(x) for large |x|.
// Returns exp(|x|) / sqrt(2*pi*|x|) * P(1/|x|)
inline vFloat calculate_i0_asymptotic_(const vFloat abs_x) {
#ifdef INP_FLOAT32
    const vFloat exp_abs = _sfpu_exp_fp32_accurate_unsafe_(abs_x);
#else
    const vFloat exp_abs = _sfpu_exp_21f_bf16_unsafe_<true>(abs_x);
#endif

    // 1/sqrt(|x|) via Quake-style magic constant + two Newton refinements.
    const vInt rsqrt_i = as<vInt>(as<vUInt>(abs_x) >> 1);
    vFloat rsqrt_y = as<vFloat>(vInt(0x5f1110a0) - rsqrt_i);
    vFloat c0 = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * (vFloat(2.2825186f) + c0 * (vFloat(2.2533049f) + c0));
    c0 = 1.0f + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = c0 * addexp(rsqrt_y, -1) + rsqrt_y;

    // 1/|x| = (1/√|x|)²
    const vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    // P(y), degree-5 minimax fit on y ∈ [1/88.5, 0.1]; max rel err ~1e-9.
    const vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894228040e-01f,
        4.9867785050e-02f,
        2.8050629091e-02f,
        2.9219405303e-02f,
        4.4742214370e-02f,
        9.0602984099e-02f);

    return exp_abs * rsqrt_y * correction;
}

template <bool APPROXIMATION_MODE>
inline void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    constexpr float I0_MAX_INPUT = 88.5f;
    constexpr float I0_THRESHOLD = 10.0f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        
        // Clamp to [-88.5, 88.5]
        x = symmetric_clamp(x, I0_MAX_INPUT);
        
        const vFloat abs_x = abs(x);
        vFloat result;

        // Polynomial path for |x| <= 10
        {
            vFloat t = x * x;
            result = 1.0f + POLYVAL10(
                                1.50E-22f,
                                7.24E-20f,
                                2.90E-17f,
                                9.39E-15f,
                                2.40E-12f,
                                4.71E-10f,
                                6.78E-08f,
                                0.000006781684028f,
                                0.0004340277778f,
                                0.015625f,
                                0.25f,
                                t);
        }

        // Asymptotic overwrite for |x| > 10
        v_if(abs_x > I0_THRESHOLD) {
            result = calculate_i0_asymptotic_(abs_x);
        }
        v_endif;

#ifndef INP_FLOAT32
        result = convert<vFloat16b>(result, RoundMode::Nearest);
#endif
        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
