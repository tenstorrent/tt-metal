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

// ======================================================================
// i0(x) — modified Bessel function of the first kind, order 0.
//
// Two-region implementation, exploiting that i0 is even: i0(-x) = i0(x).
//   |x| <= 10:  rational p(t)/q(t) on t = x^2
//               FP32: 8 numer + 9 denom coeffs (minimax), max rel err 1.8e-09
//               BF16: 4 numer + 4 denom coeffs (minimax), max rel err 5.1e-04
//   |x| > 10:  asymptotic expansion
//                 i0(x) = exp(|x|) / sqrt(|x|) * C(1/|x|)
//               degree-7 Chebyshev fit of the A&Stegun asymptotic C(y);
//               1/sqrt(2*pi) folded into the leading coefficient;
//               max rel err (math model, asymptotic region) 1.6e-11
//
// Code shape is identical to ckernel_sfpu_i1.h: the rational path is
// computed unconditionally, then v_if(|x|>10) overwrites DST with the
// asymptotic result, which relieves SFPI LRA pressure.
//
// Inputs are clamped to [-88.5, 88.5] to avoid exp() overflow.
//
// APPROXIMATION_MODE: only affects the reciprocal NR iteration count.
// ======================================================================


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

    // P(y), degree-5 minimax fit on y ∈ [1/88.5, 0.1]; max rel err ~1e-9.
    // This outlined function does not stress the main loop's LRA, so full precision is safe.
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894227974021e-01f,
        4.9867938148768e-02f,
        2.8036736869777e-02f,
        2.9865842464096e-02f,
        2.7845939002042e-02f,
        3.4306861607701e-01f,
        -1.8179118631608e+00f,
        8.0219616941762e+00f);;

    // i1 is odd: copy sign of original x onto positive magnitude.
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

        sfpi::vFloat val;
        // ─── Polynomial path (always; valid for |x| ≤ 10) ────────────────
        // Computed unconditionally and stored — its LRegs are then free
        // for the asymptotic block to use.
        {
            const sfpi::vFloat t = x * x;
#ifdef INP_FLOAT32
            sfpi::vFloat numer = PolynomialEvaluator::eval(
                t,
                1.000000000000e+00f,
                2.483673425852e-01f,
                1.529958418282e-02f,
                4.275682543689e-04f,
                6.970370070201e-06f,
                7.043065448874e-08f,
                4.073547521013e-10f,
                1.081852696115e-12f);;
            sfpi::vFloat denom = PolynomialEvaluator::eval(
                t,
                1.000000000000e+00f,
                -1.632658102755e-03f,
                8.274955077675e-05f,
                -1.636957468207e-06f,
                1.363852406505e-08f,
                -6.657140956072e-11f,
                2.056318206039e-13f,
                -3.821679840770e-16f,
                3.317315978850e-19f);;
#else
            sfpi::vFloat numer = PolynomialEvaluator::eval(
                t, 9.994889398e-01f,
                2.327333338e-01f,
                1.079707336e-02f,
                2.046399766e-04f);;
            sfpi::vFloat denom =
                PolynomialEvaluator::eval(t, 1.000000000e+00f,
                -1.848503178e-02f,
                1.324594034e-04f,
                -3.565079939e-07f);;
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
