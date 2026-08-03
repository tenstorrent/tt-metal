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

namespace ckernel::sfpu {

// ======================================================================
// i0(x) — modified Bessel function of the first kind, order 0.
//
// Two-region implementation, exploiting that i0 is even: i0(-x) = i0(x).
//   |x| ≤ 10:  Maclaurin series on t = x² (same polynomial as before,
//              accurate to ~0.5 ULP in this range).
//   |x| > 10:  asymptotic expansion
//                i0(x) = exp(|x|) / sqrt(|x|) · P(1/|x|)
//              degree-5 minimax fit (6 coeffs), max rel err ~1e-9 over [10, 88.5].
//
// Code shape (matching i1's pattern to stay within SFPI LRA budget):
//   1. Compute polynomial result unconditionally and store to DST.
//      Polynomial-path intermediates die at the store, freeing LRegs.
//   2. v_if (|x|>10): overwrite DST with asymptotic result.
//
// Inputs are clamped to [-88.5, 88.5] to avoid exp() overflow.
// ======================================================================

// Asymptotic path — outlined to keep register pressure within SFPI's
// LRA budget. Returns exp(|x|) · 1/√(|x|) · P(1/|x|).
// i0 is even, so no sign handling needed (cf. i1's copysgn).
inline sfpi::vFloat calculate_i0_asymptotic_(const sfpi::vFloat abs_x) {
    // exp(|x|) — unsafe variants: |x|∈[10,88.5] precludes
    // overflow/underflow, so safe wrappers' clamping/guards are dead.
#ifdef INP_FLOAT32
    const sfpi::vFloat exp_abs = _sfpu_exp_fp32_accurate_unsafe_(abs_x);
#else
    const sfpi::vFloat exp_abs = _sfpu_exp_21f_bf16_unsafe_<true>(abs_x);
#endif

    // 1/sqrt(|x|) via Quake-style magic constant + two Newton refinements.
    // Computed first so that 1/|x| can be derived as rsqrt_y² without a
    // separate sfpu_reciprocal call.
    const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
    sfpi::vFloat rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - rsqrt_i);
    sfpi::vFloat c0 = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * (sfpi::vFloat(2.2825186f) + c0 * (sfpi::vFloat(2.2533049f) + c0));
    c0 = 1.0f + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;

    // 1/|x| = (1/√|x|)² — reuses the refined rsqrt instead of a fresh reciprocal.
    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    // P(y), degree-5 minimax fit on y ∈ [1/88.5, 0.1]; max rel err ~1e-9.
    // Leading coefficient 1/√(2π) ≈ 0.3989422804 scaled into the fit.
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894228040e-01f,
        4.9867785050e-02f,
        2.8050629300e-02f,
        2.9219405000e-02f,
        3.2787563000e-02f,
        3.9305493000e-02f);

    return exp_abs * rsqrt_y * correction;
}

// Original Maclaurin polynomial — preserved for |x| ≤ 10 where it is accurate.
// Uses Horner form in t = x². I0(x) = 1 + x²/4 + x⁴/64 + x⁶/2304 + x⁸/147456 + ...
#define I0_POLYVAL10(coef10, coef9, coef8, coef7, coef6, coef5, coef4, coef3, coef2, coef1, coef0, t4) \
    ((coef0 +                                                                                          \
      (coef1 +                                                                                         \
       (coef2 +                                                                                        \
        (coef3 +                                                                                       \
         (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) *  \
              t4) *                                                                                    \
             t4) *                                                                                     \
            t4) *                                                                                      \
           t4) *                                                                                       \
          t4) *                                                                                        \
     t4)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    constexpr float I0_MAX_INPUT = 88.5f;
    constexpr float I0_THRESHOLD = 10.0f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Clamp to [-88.5, 88.5] — exp() saturates near ±88.7 in FP32.
        x = sfpi::symmetric_clamp(x, I0_MAX_INPUT);

        const sfpi::vFloat abs_x = sfpi::abs(x);

        sfpi::vFloat val;
        // ─── Polynomial path (always; valid for |x| ≤ 10) ────────────────
        // Computed unconditionally and stored — its LRegs are then free
        // for the asymptotic block to use.
        {
            const sfpi::vFloat t = x * x;
            val = 1.0f + I0_POLYVAL10(
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

        // ─── Asymptotic overwrite for OOD lanes (|x| > 10) ───────────────
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

}  // namespace ckernel::sfpu
