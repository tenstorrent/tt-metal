// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// i1(x) — modified Bessel function of the first kind, order 1.
//
// Two-region implementation, exploiting that i1 is odd: i1(-x) = -i1(x).
//   |x| ≤ 10:  rational p(t)/q(t) on t = x², result = x · p(t)/q(t)
//              BF16: 4 numer + 4 denom coeffs in t (= n7/d6 in x) → 0.02 BF16 ULP analytical
//              FP32: 7 numer + 8 denom coeffs in t (= n13/d14 in x) → <0.001 FP32 ULP analytical
//   |x| > 10:  asymptotic expansion
//                i1(x) = sign(x) · exp(|x|) / sqrt(|x|) · P(1/|x|)
//              degree-5 minimax fit (6 coeffs), max rel err ~1e-9 over [10, 88.5].
//
// Code shape (chosen to relieve SFPI LRA budget):
//   1. Compute polynomial result unconditionally and store to DST.
//      Polynomial-path intermediates die at the store, freeing LRegs.
//   2. v_if (|x|>10): overwrite DST with asymptotic result.
// This is semantically identical to a v_if/v_else split but lets the
// register allocator schedule the two paths sequentially rather than
// keeping the polynomial alive across the asymptotic block.
//
// Inputs are clamped to [-88.5, 88.5] to avoid exp() overflow.
// In-domain accuracy is unchanged from the polynomial-only baseline.
// OOD accuracy: ~10⁶ FP32 ULP (clamping) → <60 FP32 ULP (asymptotic with
// accurate FP32 exp).
//
// APPROXIMATION_MODE: only affects the reciprocal NR iteration count.
// ======================================================================

// Asymptotic path is outlined to keep register pressure within SFPI's
// LRA budget. Returns sign(x_signed) · exp(|x|) · 1/sqrt(|x|) · P(1/|x|).
// Note: this function must stay minimalist — SFPU LRA is limited.
// Every operation here competes with the main loop.
inline sfpi::vFloat calculate_i1_asymptotic_(const sfpi::vFloat abs_x, const sfpi::vFloat x_signed) {
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
    const sfpi::vInt rsqrt_i = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(abs_x) >> 1);
    sfpi::vFloat rsqrt_y = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - rsqrt_i);
    sfpi::vFloat c0 = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * (sfpi::vFloat(2.2825186f) + c0 * (sfpi::vFloat(2.2533049f) + c0));
    c0 = sfpi::vConst1 + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;

    // 1/|x| = (1/√|x|)² — reuses the refined rsqrt instead of a fresh reciprocal.
    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    // P(y), degree-5 minimax fit on y ∈ [1/88.5, 0.1]; max rel err ~1e-9.
    // This outlined function does not stress the main loop's LRA, so full precision is safe.
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894228967e-01f,
        -1.4960495444e-01f,
        -4.6652925320e-02f,
        -4.3674591560e-02f,
        -1.9748322314e-02f,
        -3.3467922914e-01f);

    // i1 is odd: copy sign of original x onto positive magnitude.
    return sfpi::copysgn(exp_abs * rsqrt_y * correction, x_signed);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i1() {
    constexpr float I1_MAX_INPUT = 88.5f;
    constexpr float I1_THRESHOLD = 10.0f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Clamp to [-88.5, 88.5] — exp() saturates near ±88.7 in FP32.
        sfpi::vFloat lo = -I1_MAX_INPUT;
        sfpi::vec_min_max(lo, x);
        sfpi::vFloat hi = I1_MAX_INPUT;
        sfpi::vec_min_max(x, hi);

        const sfpi::vFloat abs_x = sfpi::setsgn(x, 0);

        sfpi::vFloat val;
        // ─── Polynomial path (always; valid for |x| ≤ 10) ────────────────
        // Computed unconditionally and stored — its LRegs are then free
        // for the asymptotic block to use.
        {
            const sfpi::vFloat t = x * x;
#ifdef INP_FLOAT32
            sfpi::vFloat numer = PolynomialEvaluator::eval(
                t,
                5.0000000000e-01f,
                5.6819390506e-02f,
                1.9247245509e-03f,
                2.8397364076e-05f,
                2.0916867527e-07f,
                7.7937084564e-10f,
                1.2293555930e-12f);
            sfpi::vFloat denom = PolynomialEvaluator::eval(
                t,
                sfpi::vConst1,
                -1.1361218989e-02f,
                6.1268139689e-05f,
                -1.9771712800e-07f,
                3.8127551116e-10f,
                -3.1218170410e-13f,
                -3.0635529988e-16f,
                7.4301498523e-19f);
#else
            sfpi::vFloat numer = PolynomialEvaluator::eval(
                t, 4.9992737740e-01f, 5.4503594600e-02f, 1.6126291630e-03f, 2.0223499130e-05f);
            sfpi::vFloat denom =
                PolynomialEvaluator::eval(t, sfpi::vConst1, -1.6242591070e-02f, 1.0333660750e-04f, -2.5076132990e-07f);
#endif
            val = numer * x * sfpu_reciprocal<APPROXIMATION_MODE>(denom);
        }

        // ─── Asymptotic overwrite for OOD lanes (|x| > 10) ───────────────
        v_if(abs_x > I1_THRESHOLD) { val = calculate_i1_asymptotic_(abs_x, x); }
        v_endif;
#ifndef INP_FLOAT32
        val = sfpi::convert<sfpi::vFloat16b>(val, sfpi::RoundMode::Nearest);
#endif
        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void i1_init() {
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
