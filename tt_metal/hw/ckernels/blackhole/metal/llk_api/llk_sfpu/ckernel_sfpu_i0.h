// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

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

// Asymptotic path for |x| > 10.0 (Abramowitz & Stegun 9.7.1)
// I0(x) = exp(|x|) / sqrt(|x|) * P(1/|x|)
inline sfpi::vFloat calculate_i0_asymptotic_(const sfpi::vFloat abs_x) {
#ifdef INP_FLOAT32
    const sfpi::vFloat exp_abs = _sfpu_exp_fp32_accurate_unsafe_(abs_x);
#else
    const sfpi::vFloat exp_abs = _sfpu_exp_21f_bf16_unsafe_<true>(abs_x);
#endif

    const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
    sfpi::vFloat rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - rsqrt_i);
    sfpi::vFloat c0 = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * (sfpi::vFloat(2.2825186f) + c0 * (sfpi::vFloat(2.2533049f) + c0));
    c0 = 1.0f + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;

    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894228967e-01f,
        4.9867785050e-02f,
        2.8050628966e-02f,
        2.9219405173e-02f,
        4.5655320582e-02f,
        1.0506316279e-01f);

    return exp_abs * rsqrt_y * correction;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    constexpr float I0_MAX_INPUT = 88.5f;
    constexpr float I0_THRESHOLD = 10.0f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat input = sfpi::dst_reg[0];

        // Clamp to [-88.5, 88.5] to avoid exp() overflow on asymptotic branch
        input = sfpi::symmetric_clamp(input, I0_MAX_INPUT);

        const sfpi::vFloat abs_x = sfpi::abs(input);
        const sfpi::vFloat x2 = input * input;

        // 1. Polynomial path (valid for |x| <= 10)
        sfpi::vFloat result = 1.0f + POLYVAL10(
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
                            x2);

        // 2. Asymptotic overwrite for out-of-domain lanes (|x| > 10)
        v_if(abs_x > I0_THRESHOLD) {
            result = calculate_i0_asymptotic_(abs_x);
        }
        v_endif;

#ifndef INP_FLOAT32
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
#endif

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE = false>
inline void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
