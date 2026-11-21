// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"
#include "ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// Note: this function must stay minimalist — SFPU LRA is limited.
// Every operation here competes with the main loop.
inline sfpi::vFloat calculate_i1_asymptotic_(
    const sfpi::vFloat rsqrt_y, const sfpi::vFloat excess, const sfpi::vFloat x_signed) {
#ifdef INP_FLOAT32
    const sfpi::vFloat exp_abs = _sfpu_exp_fp32_accurate_unsafe_(excess);
#else
    const sfpi::vFloat exp_abs = _sfpu_exp_21f_bf16_unsafe_<true>(excess);
#endif

    // 1/|x| = (1/√|x|)² — reuses the refined rsqrt instead of a fresh reciprocal.
    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    // This outlined function does not stress the main loop's LRA, so full precision is safe.
    const sfpi::vFloat correction = PolynomialEvaluator::eval(
        inv_abs_x,
        7.754754883e+03f,
        -2.908041260e+03f,
        -9.078535767e+02f,
        -8.297672119e+02f,
        -5.510595703e+02f,
        -5.964537109e+03f);

    // i1 is odd: copy sign of original x onto positive magnitude.
    return sfpi::copysgn(exp_abs * rsqrt_y * correction, x_signed);
}

inline sfpi::vFloat calculate_i1_rsqrt_(const sfpi::vFloat abs_x) {
    const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
    sfpi::vFloat rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - rsqrt_i);
    sfpi::vFloat c0 = (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = rsqrt_y * (sfpi::vFloat(2.2825186f) + c0 * (sfpi::vFloat(2.2533049f) + c0));
    c0 = 1.0f + (-rsqrt_y) * (abs_x * rsqrt_y);
    rsqrt_y = c0 * sfpi::addexp(rsqrt_y, -1) + rsqrt_y;
    return rsqrt_y;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i1() {
    constexpr float I1_MAX_INPUT = 92.0f;
    constexpr float I1_THRESHOLD = 9.875f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

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

        const sfpi::vFloat excess = abs_x - I1_THRESHOLD;
        const sfpi::vFloat rsqrt_y = calculate_i1_rsqrt_(abs_x);
        v_if(excess >= 0.0f) { val = calculate_i1_asymptotic_(rsqrt_y, excess, x); }
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
