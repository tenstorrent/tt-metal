// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_bessel_common.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
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
//              degree-5 minimax fit (6 coeffs), max rel err ~1e-9 over [10, 92].
//
// Code shape (chosen to relieve SFPI LRA budget):
//   1. Compute polynomial result unconditionally and store to DST.
//      Polynomial-path intermediates die at the store, freeing LRegs.
//   2. v_if (|x|>10): overwrite DST with asymptotic result.
// This is semantically identical to a v_if/v_else split but lets the
// register allocator schedule the two paths sequentially rather than
// keeping the polynomial alive across the asymptotic block.
//
// Inputs are clamped to [-92, 92], which is i1's own FP32 overflow point
// (91.90626) rounded up, not exp's (88.72284). The two differ because the
// asymptotic value carries a 1/sqrt(2·pi·|x|) ≈ 1/23.6 divisor, so i1 stays
// representable for ln(sqrt(2·pi·88.5)) = 3.16 more of domain than the bare
// exp(|x|) intermediate does. The intermediate is kept in range by evaluating
// exp(|x|)/2^32 and rescaling at the end (see calculate_i1_asymptotic_), so the
// only operation that can overflow is that final rescale — and overflowing
// there is the correct answer, because that is where i1 itself leaves FP32.
// Every |x| > 92, including +/-Inf, therefore lands on +/-Inf rather than on
// the finite value of i1 at the clamp point.
// In-domain accuracy is unchanged from the polynomial-only baseline.
// OOD accuracy: ~10⁶ FP32 ULP (clamping) → <60 FP32 ULP (asymptotic with
// accurate FP32 exp).
//
// APPROXIMATION_MODE: only affects the reciprocal NR iteration count.
// ======================================================================

// Asymptotic path: shared machinery lives in ckernel_sfpu_bessel_common.h; this
// wrapper carries P's coefficients (fit on y ∈ [1/92, 0.1], max rel err 1.052e-9
// in float64), the 2^32 rescale folded into each coefficient (exact power of two,
// so the emitted constants change but no mantissa does), and the final sign
// fix-up — i1 is odd, i0 is not.
//
// exp_abs · rsqrt_y peaks at 2.2e29 for |x|=92, so the rescaled polynomial
// multiply is the only operation here that can overflow — which is correct,
// because that is where i1 itself leaves FP32.
inline sfpi::vFloat calculate_i1_asymptotic_(const sfpi::vFloat abs_x, const sfpi::vFloat x_signed) {
#ifdef INP_FLOAT32
    const sfpi::vFloat mag = _bessel_asymptotic_<true>(
#else
    const sfpi::vFloat mag = _bessel_asymptotic_<false>(
#endif
        abs_x,
        3.9894228967e-01f,
        -1.4960495444e-01f,
        -4.6652925320e-02f,
        -4.3674591560e-02f,
        -1.9748322314e-02f,
        -3.3467922914e-01f);
    return sfpi::copysgn(mag, x_signed);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i1() {
    // i1, not exp: i1(x) leaves FP32 at |x| = 91.90626, exp(x) at 88.72284.
    // Clamping at the larger bound keeps the 3.16 of domain in between, on which
    // i1 is finite and representable, and sends everything above it to +/-Inf,
    // which is the correct saturation for a function that has left the format.
    constexpr float I1_MAX_INPUT = 92.0f;
    constexpr float I1_THRESHOLD = 10.0f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Clamp to [-92, 92] — past that i1 has overflowed FP32, so the clamped
        // input reaches the asymptotic path and overflows to ±Inf on its own.
        x = sfpi::symmetric_clamp(x, I1_MAX_INPUT);

        const sfpi::vFloat abs_x = sfpi::abs(x);

        // ─── Polynomial path (always; valid for |x| ≤ 10) ────────────────
        // Computed unconditionally and stored — its LRegs are then free
        // for the asymptotic block to use.
        sfpi::vFloat val;
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
                1.0f,
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
                PolynomialEvaluator::eval(t, 1.0f, -1.6242591070e-02f, 1.0333660750e-04f, -2.5076132990e-07f);
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
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
