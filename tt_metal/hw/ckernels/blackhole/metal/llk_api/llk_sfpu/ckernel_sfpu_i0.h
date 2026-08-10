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
//   |x| ≤ 10:  rational p(t)/q(t) on t = x², i0(x) = p(t)/q(t)
//              FP32: 7 numer + 7 denom coeffs in t (= n12/d12 in x)
//                    → ~3.7e-6 max relative error over [-10,10] in float32
//   |x| > 10:  asymptotic expansion
//                i0(x) = exp(|x|) / sqrt(2·π·|x|) · P(1/|x|)
//              degree-5 minimax fit (6 coeffs), max rel err ~2e-9 over
//              [10, 88.5]; the leading coefficients reproduce the standard
//              series (1 + 1/(8x) + 9/(128x²) + …).
//
// This replaces the previous single fixed 12-term Maclaurin series, which
// had no domain split and silently produced wrong results past |x|≈13
// (21% rel. error at x=20, 89% at x=30). It mirrors the sibling i1 kernel
// (PR #43246), which already carried the two-region fix.
//
// Code shape (chosen to relieve SFPI LRA budget, as in i1):
//   1. Compute the polynomial result unconditionally and store to DST.
//      Polynomial-path intermediates die at the store, freeing LRegs.
//   2. v_if (|x|>10): overwrite DST with the asymptotic result.
// This is semantically identical to a v_if/v_else split but lets the
// register allocator schedule the two paths sequentially rather than
// keeping the polynomial alive across the asymptotic block.
//
// Inputs are clamped to [-88.5, 88.5] to avoid exp() overflow.
// APPROXIMATION_MODE: only affects the reciprocal NR iteration count.
// ======================================================================

// Asymptotic path is outlined to keep register pressure within SFPI's
// LRA budget. Returns exp(|x|) · 1/sqrt(|x|) · P(1/|x|); i0 is even so the
// result is positive regardless of the sign of x.
inline sfpi::vFloat calculate_i0_asymptotic_(const sfpi::vFloat abs_x) {
    // exp(|x|) — unsafe variants in both paths: |x|∈[10,88.5] precludes
    // overflow/underflow, so the safe wrappers' clamping/guards are dead
    // and skipped.
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

    // P(y) · 1/√(2π), degree-5 minimax fit on y ∈ [1/88.5, 0.1]; the kernel
    // computes exp(|x|)/√|x| · correction, so the 1/√(2π) is folded here and
    // the result equals exp(|x|)/√(2π|x|) · P_true(1/|x|). Max rel err ~2e-9.
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894228488e-01f,
        4.9865880863e-02f,
        2.7962177078e-02f,
        3.1670128495e-02f,
        1.1924552096e-02f,
        2.8334664358e-01f);

    return exp_abs * rsqrt_y * correction;
}

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

        // ─── Polynomial path (always; valid for |x| ≤ 10) ────────────────
        // Computed unconditionally and stored — its LRegs are then free
        // for the asymptotic block to use. i0 is even, so the rational is
        // p(t)/q(t) with t = x² (no leading x factor, unlike i1).
        sfpi::vFloat val;
        {
            const sfpi::vFloat t = x * x;
#ifdef INP_FLOAT32
            sfpi::vFloat numer = PolynomialEvaluator::eval(
                t,
                9.9999998049e-01f,
                2.2739165401e-01f,
                1.0191804746e-02f,
                1.3427654871e-04f,
                8.4735216695e-08f,
                -8.6757469247e-09f,
                -4.8694484833e-11f);
            sfpi::vFloat denom = PolynomialEvaluator::eval(
                t,
                1.0f,
                -2.2608410615e-02f,
                2.1894030161e-04f,
                -1.2360964095e-06f,
                4.3894063778e-09f,
                -9.3473138462e-12f,
                9.2859575595e-15f);
#else
            // BF16/FP8 path: a lighter 5/5 rational keeps the same threshold.
            sfpi::vFloat numer = PolynomialEvaluator::eval(
                t,
                1.0000398695e+00f,
                2.3368603245e-01f,
                1.1698542439e-02f,
                2.0719575997e-04f,
                1.5218651330e-06f);
            sfpi::vFloat denom = PolynomialEvaluator::eval(
                t,
                1.0f,
                -1.6244970065e-02f,
                1.1661809746e-04f,
                -4.3240631926e-07f,
                6.8558653470e-10f);
#endif
            val = numer * sfpu_reciprocal<APPROXIMATION_MODE>(denom);
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
