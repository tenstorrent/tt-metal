// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// i0(x) — modified Bessel function of the first kind, order 0.
//
// Two-region implementation, exploiting that i0 is even: i0(-x) = i0(x).
//   |x| ≤ 10:  polynomial Maclaurin fit on t = x²
//   |x| > 10:  asymptotic expansion i0(x) = exp(|x|) / sqrt(|x|) · P(1/|x|)
//              degree-5 minimax fit (6 coeffs), max rel err ~1.97e-9 over [10, 88.5].
//
// Inputs are clamped to [-88.5, 88.5] to avoid exp() overflow.
// ======================================================================

inline sfpi::vFloat calculate_i0_asymptotic_(const sfpi::vFloat abs_x) {
#ifdef INP_FLOAT32
    const sfpi::vFloat exp_abs = _sfpu_exp_fp32_accurate_unsafe_(abs_x);
#else
    const sfpi::vFloat exp_abs = _sfpu_exp_21f_bf16_unsafe_<true>(abs_x);
#endif

    // 1/sqrt(|x|) via magic constant + Newton refinement
    const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
    sfpi::vFloat rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - rsqrt_i);
    sfpi::vFloat c0 = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * (sfpi::vFloat(2.2825186f) + c0 * (sfpi::vFloat(2.2533049f) + c0));
    c0 = sfpi::vConst1 + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;

    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894227094e-01f,
        4.9869311030e-02f,
        2.7961319185e-02f,
        3.1664597617e-02f,
        1.1930477377e-02f,
        2.8326601569e-01f);

    return exp_abs * rsqrt_y * correction;
}

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

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    constexpr float I0_MAX_INPUT = 88.5f;
    constexpr float I0_THRESHOLD = 10.0f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Clamp to [-88.5, 88.5]
        sfpi::vFloat lo = -I0_MAX_INPUT;
        sfpi::vec_min_max(lo, x);
        sfpi::vFloat hi = I0_MAX_INPUT;
        sfpi::vec_min_max(x, hi);

        const sfpi::vFloat abs_x = sfpi::setsgn(x, 0);

        sfpi::vFloat val;
        {
            const sfpi::vFloat t = x * x;
            val = 1.0f + POLYVAL10(
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

        sfpi::v_if (abs_x > I0_THRESHOLD) {
            val = calculate_i0_asymptotic_(abs_x);
        }
        sfpi::v_endif;

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
