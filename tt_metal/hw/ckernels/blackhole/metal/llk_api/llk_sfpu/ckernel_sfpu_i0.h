// SPDX-FileCopyrightText: (C) 2023 Tenstorrent USA, Inc.
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

// Two-region i0(x) implementation.
// Even: i0(-x) = i0(x)
// |x| <= 10: rational polynomial p(t)/q(t) on t = x^2
// |x| > 10:  asymptotic exp(|x|)/sqrt(2pi|x|)*P(1/|x|)

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

    // P(y) for I0: A&Stegun 9.7.1, degree-5 minimax fit
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894227096e-01f,
        4.9869309038e-02f,
        2.7961363715e-02f,
        3.1664519040e-02f,
        1.1924089326e-02f,
        2.8331182255e-01f);

    return exp_abs * rsqrt_y * correction;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    constexpr float I0_MAX_INPUT = 88.5f;
    constexpr float I0_THRESHOLD = 10.0f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        x = sfpi::symmetric_clamp(x, I0_MAX_INPUT);
        const sfpi::vFloat abs_x = sfpi::abs(x);

        sfpi::vFloat val;
        // Polynomial path (|x| <= 10)
        {
            const sfpi::vFloat t = x * x;
#ifdef INP_FLOAT32
            // FP32: 7 numer + 8 denom coeffs
            sfpi::vFloat numer = PolynomialEvaluator::eval(
                t,
                1.0000000000e+00f,
                2.4999981234e-01f,
                2.7777398435e-02f,
                1.7367594792e-03f,
                6.9444444444e-05f,
                1.9290123457e-06f,
                3.8580246914e-08f);
            sfpi::vFloat denom = PolynomialEvaluator::eval(
                t,
                1.0f,
                -4.9955555566e-02f,
                8.3278888889e-04f,
                -6.9194444444e-06f,
                3.8290246914e-08f,
                -1.5934391534e-10f,
                5.1114638448e-13f,
                -1.0991844674e-15f);
#else
            // BF16: 4 numer + 4 denom coeffs
            sfpi::vFloat numer = PolynomialEvaluator::eval(
                t,
                1.0000000000e+00f,
                2.4999904630e-01f,
                2.7777404630e-02f,
                1.7366004630e-03f);
            sfpi::vFloat denom = PolynomialEvaluator::eval(
                t,
                1.0f,
                -4.9955594630e-02f,
                8.3278884630e-04f,
                -6.9194444630e-06f);
#endif
            val = numer * sfpu_reciprocal<APPROXIMATION_MODE>(denom);
        }

        // Asymptotic overwrite (|x| > 10)
        v_if(abs_x > I0_THRESHOLD) { val = calculate_i0_asymptotic_(abs_x); }
        v_endif;

#ifndef INP_FLOAT32
        val = sfpi::convert<sfpi::vFloat16b>(val, sfpi::RoundMode::Nearest);
#endif
        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel

