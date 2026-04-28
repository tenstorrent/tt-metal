// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// i1(x) = x * p(x²) / q(x²)  — odd function, evaluated in t = x²
//
// BF16: n7/d6  → degree-3 in t, 0.02 BF16 ULP analytical
// FP32: n14/d14 → degree-6/7 in t, <0.001 FP32 ULP analytical
//        (arithmetic-limited to ~8 ULP on SFPU due to FP32 rounding)
// APPROXIMATION_MODE: approx adds ~128 FP32 ULP via sfpu_reciprocal
//   (negligible for BF16 ≤1 ULP, degrades FP32 from ~11 to ~140 ULP)
// ======================================================================

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i1() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Clamp to [-10, 10]: polynomial extrapolates catastrophically for |x| > 12.
        sfpi::vFloat lo = -1.0000000000e+01f;
        sfpi::vec_min_max(lo, x);
        sfpi::vFloat hi = 1.0000000000e+01f;
        sfpi::vec_min_max(x, hi);

        const sfpi::vFloat t = x * x;  // t = x²

#ifdef INP_FLOAT32
        // FP32 n14/d14: p degree-6, q degree-7 in t.
        // Coefficients in ascending order of t^k (non-zero parity terms only).
        sfpi::vFloat numer = PolynomialEvaluator::eval(
            t,
            5.0000000000e-01f,   // a1  (t^0)
            5.6819390506e-02f,   // a3  (t^1)
            1.9247245509e-03f,   // a5  (t^2)
            2.8397364076e-05f,   // a7  (t^3)
            2.0916867527e-07f,   // a9  (t^4)
            7.7937084564e-10f,   // a11 (t^5)
            1.2293555930e-12f);  // a13 (t^6)
        sfpi::vFloat denom = PolynomialEvaluator::eval(
            t,
            1.0000000000e+00f,   // b0  (t^0)
            -1.1361218989e-02f,  // b2  (t^1)
            6.1268139689e-05f,   // b4  (t^2)
            -1.9771712800e-07f,  // b6  (t^3)
            3.8127551116e-10f,   // b8  (t^4)
            -3.1218170410e-13f,  // b10 (t^5)
            -3.0635529988e-16f,  // b12 (t^6)
            7.4301498523e-19f);  // b14 (t^7)
#else
        // BF16 n7/d6: p degree-3, q degree-3 in t.
        // Minimax-fitted via Chebyshev-node differential evolution.
        sfpi::vFloat numer = PolynomialEvaluator::eval(
            t,
            4.9992737740e-01f,   // a1 (t^0)
            5.4503594600e-02f,   // a3 (t^1)
            1.6126291630e-03f,   // a5 (t^2)
            2.0223499130e-05f);  // a7 (t^3)
        sfpi::vFloat denom = PolynomialEvaluator::eval(
            t,
            1.0000000000e+00f,    // b0 (t^0)
            -1.6242591070e-02f,   // b2 (t^1)
            1.0333660750e-04f,    // b4 (t^2)
            -2.5076132990e-07f);  // b6 (t^3)
#endif

        sfpi::dst_reg[0] = numer * x * sfpu_reciprocal<APPROXIMATION_MODE>(denom);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void i1_init() {
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
