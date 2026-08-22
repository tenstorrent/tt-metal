// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// i0(x) — modified Bessel function of the first kind, order 0.
//
// Two-region implementation, exploiting that i0 is even: i0(-x) = i0(x).
//   |x| ≤ 7:   Maclaurin polynomial (12 terms, same as before)
//              max rel err ~3.2e-07 over [0, 7]
//   |x| >  7:  asymptotic expansion (Abramowitz & Stegun 9.7.1)
//                i0(x) = exp(|x|) / sqrt(2π|x|) · P(1/|x|)
//              degree-5 asymptotic series in 1/|x|, max rel err ~6e-06
//
// Code shape (chosen to relieve SFPI LRA budget):
//   1. Compute polynomial result unconditionally.
//      Polynomial-path intermediates die at the store, freeing LRegs.
//   2. v_if (|x|>7): overwrite with asymptotic result.
// This is semantically identical to a v_if/v_else split but lets the
// register allocator schedule the two paths sequentially.
//
// Inputs are clamped to [-88.5, 88.5] to avoid exp() overflow.
// Overall worst-case rel err ~6e-06 over [0, 88.5], well within
// 1 BF16 ULP (~4e-3).
//
// APPROXIMATION_MODE: used by the templated calculate_i0() for
// SFPI approximation flags.
// ======================================================================

// Asymptotic path outlined to keep register pressure within SFPI's LRA budget.
// Returns exp(|x|) / sqrt(2π|x|) · P(1/|x|).
// Note: i0 is even — no sign handling needed.
inline sfpi::vFloat calculate_i0_asymptotic_(const sfpi::vFloat abs_x) {
    // exp(|x|) — unsafe variants: |x|∈[7,88.5] precludes overflow/underflow.
#ifdef INP_FLOAT32
    const sfpi::vFloat exp_abs = _sfpu_exp_fp32_accurate_unsafe_(abs_x);
#else
    const sfpi::vFloat exp_abs = _sfpu_exp_21f_bf16_unsafe_<true>(abs_x);
#endif

    // 1/sqrt(|x|) via Quake-style magic constant + two Newton refinements.
    // Computed first so that 1/|x| can be derived as rsqrt_y².
    const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
    sfpi::vFloat rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - rsqrt_i);
    sfpi::vFloat c0 = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * (sfpi::vFloat(2.2825186f) + c0 * (sfpi::vFloat(2.2533049f) + c0));
    c0 = 1.0f + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;

    // 1/|x| = (1/√|x|)² — reuses the refined rsqrt.
    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    // P(y), degree-5 asymptotic series on y ∈ [1/88.5, 1/7]; max rel err ~6e-06.
    // Abramowitz & Stegun 9.7.1 for nu=0: S(y) = 1 + 1/8*y + 9/128*y^2 + 75/1024*y^3 + ...
    // P(y) = S(y) / sqrt(2π), so all coefficients are positive.
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894228040e-01f,
        4.9867785050e-02f,
        2.8050629091e-02f,
        2.9219405303e-02f,
        4.4742214370e-02f,
        9.0602984099e-02f);

    return exp_abs * rsqrt_y * correction;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    constexpr float I0_MAX_INPUT = 88.5f;
    constexpr float I0_THRESHOLD = 7.0f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Clamp to [-88.5, 88.5] — exp() saturates near ±88.7 in FP32.
        x = sfpi::symmetric_clamp(x, I0_MAX_INPUT);

        const sfpi::vFloat abs_x = sfpi::abs(x);

        sfpi::vFloat val;
        // ─── Polynomial path (always; valid for |x| ≤ 7) ─────────────────
        // Maclaurin series: i0(x) = sum_{k=0}^{11} (x²/4)^k / (k!)²
        // Computed unconditionally — intermediates die after the store.
        {
            val = PolynomialEvaluator::eval(
                x * x,
                1.0f,
                2.5000000000e-01f,
                1.5625000000e-02f,
                4.3402777780e-04f,
                6.7816840280e-06f,
                6.7816840280e-08f,
                4.7100000000e-10f,
                2.4000000000e-12f,
                9.3900000000e-15f,
                2.9000000000e-17f,
                7.2400000000e-20f,
                1.5000000000e-22f);
        }

        // ─── Asymptotic overwrite for OOD lanes (|x| > 7) ─────────────────
        v_if(abs_x > I0_THRESHOLD) { val = calculate_i0_asymptotic_(abs_x); }
        v_endif;
#ifndef INP_FLOAT32
        val = sfpi::convert<sfpi::vFloat16b>(val, sfpi::RoundMode::Nearest);
#endif
        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
}

}  // namespace ckernel::sfpu
