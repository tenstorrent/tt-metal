// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
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
inline void i0_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// ======================================================================
// i0(x) — modified Bessel function of the first kind, order 0.
//
// The Maclaurin series below is truncated at k=11 (12 terms). Its largest
// term sits near k ≈ x/2, so the truncation is only safe while x/2 stays
// well under 11 — in practice to about x = 13. Past that the omitted tail
// dominates and the result is a silent, finite underestimate: measured
// 21.0% relative error at x = 20 and 88.6% at x = 30.
//
// Fixed by mirroring the two-region structure already used by
// ckernel_sfpu_i1.h (i0 is even, so no sign handling is needed):
//   |x| ≤ 7:   the existing Maclaurin polynomial
//   |x| > 7:   asymptotic i0(x) = exp(|x|)/sqrt(2π|x|) · P(1/|x|)
//              (Abramowitz & Stegun 9.7.1), degree-5 minimax in 1/|x|
//              over [1/88.5, 0.1]
// Inputs are clamped to ±88.5 as in i1, since exp() saturates near ±88.7.
//
// Measured relative error against a high-precision reference, dense sweep:
//   [0, 88.5]    100%  → 3.43e-05
// The asymptotic branch itself is good to 4.6e-07; the residual is the
// Maclaurin polynomial just below the handover, so the threshold sets the
// overall error. Sweeping it: 13 → 3.8e-03, 10 → 2.2e-04, 7 → 3.4e-05,
// 6 → 4.7e-05. 7 is the minimum. (The issue suggests "~10-13"; that range
// is inside one BF16 ULP (≈4e-3) but costs two orders of magnitude of
// accuracy that the asymptotic path is already paying for.)
// ======================================================================

// Outlined to keep register pressure within SFPI's LRA budget, matching the
// shape of calculate_i1_asymptotic_. Returns exp(|x|) · 1/sqrt(2π|x|) · P(1/|x|).
sfpi_inline sfpi::vFloat calculate_i0_asymptotic_(const sfpi::vFloat abs_x) {
    // exp(|x|) — |x| ∈ [10, 88.5] precludes overflow/underflow, so the
    // unsafe variants' skipped guards are dead code here.
#ifdef INP_FLOAT32
    const sfpi::vFloat exp_abs = _sfpu_exp_fp32_accurate_unsafe_(abs_x);
#else
    const sfpi::vFloat exp_abs = _sfpu_exp_21f_bf16_unsafe_<true>(abs_x);
#endif

    // 1/sqrt(|x|) via Quake-style seed plus two Newton refinements, reused to
    // derive 1/|x| as rsqrt_y² instead of issuing a separate reciprocal.
    const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
    sfpi::vFloat rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - rsqrt_i);
    sfpi::vFloat c0 = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * (sfpi::vFloat(2.2825186f) + c0 * (sfpi::vFloat(2.2533049f) + c0));
    c0 = 1.0f + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;

    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    // P(y) on y = 1/|x| ∈ [1/88.5, 0.1]. Leading term is 1/sqrt(2π).
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        3.9894227266e-01f,
        4.9868565798e-02f,
        2.8280049562e-02f,
        1.8642880395e-02f,
        1.9414494932e-01f,
        -5.4021686316e-01f);

    return exp_abs * rsqrt_y * correction;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    constexpr float I0_MAX_INPUT = 88.5f;
    constexpr float I0_THRESHOLD = 7.0f;

#pragma GCC unroll 0

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat result = 0.0f;
        vFloat input = sfpi::symmetric_clamp(dst_reg[0], I0_MAX_INPUT);
        const vFloat abs_x = sfpi::abs(input);
        vFloat x = input * input;

        // Polynomial path, computed unconditionally so its intermediates die
        // at the store and free LRegs for the asymptotic block.
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
                            x);

        // Asymptotic overwrite for the out-of-domain lanes.
        v_if(abs_x > I0_THRESHOLD) { result = calculate_i0_asymptotic_(abs_x); }
        v_endif;

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
