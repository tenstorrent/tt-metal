// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// ======================================================================
// i1(x) = x * p(x²) / q(x²)  — odd function, evaluated in t = x²
//
// Numerator and denominator Horner chains are interleaved to hide
// SFPU pipeline latency (independent chains, no data dependency).
//
// BF16: n7/d6  → degree-3 in t, 3 interleaved MAD pairs, 0.02 BF16 ULP
// FP32: n14/d14 → degree-6/7 in t, 7 interleaved MAD pairs, <0.001 FP32 ULP
//        (arithmetic-limited to ~8 ULP on SFPU due to FP32 rounding)
// APPROXIMATION_MODE threads into sfpu_reciprocal: approx adds ~128 FP32 ULP
//   (negligible for BF16 ≤1 ULP, but degrades FP32 from ~11 to ~140 ULP)
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
        // Interleaved Horner: independent num/den chains overlap in the pipeline.
        sfpi::vFloat n = 1.2293555930e-12f, d2 = 7.4301498523e-19f;
        n = n * t + 7.7937084564e-10f;
        d2 = d2 * t + -3.0635529988e-16f;
        n = n * t + 2.0916867527e-07f;
        d2 = d2 * t + -3.1218170410e-13f;
        n = n * t + 2.8397364076e-05f;
        d2 = d2 * t + 3.8127551116e-10f;
        n = n * t + 1.9247245509e-03f;
        d2 = d2 * t + -1.9771712800e-07f;
        n = n * t + 5.6819390506e-02f;
        d2 = d2 * t + 6.1268139689e-05f;
        n = n * t + 5.0000000000e-01f;
        d2 = d2 * t + -1.1361218989e-02f;
        /* d2 has one more term (degree-7 vs degree-6) */ d2 = d2 * t + 1.0000000000e+00f;
#else
        // BF16 n7/d6: p degree-3, q degree-3 in t.
        // Interleaved Horner: 3 MAD pairs.
        sfpi::vFloat n = 2.0223499130e-05f, d2 = -2.5076132990e-07f;
        n = n * t + 1.6126291630e-03f;
        d2 = d2 * t + 1.0333660750e-04f;
        n = n * t + 5.4503594600e-02f;
        d2 = d2 * t + -1.6242591070e-02f;
        n = n * t + 4.9992737740e-01f;
        d2 = d2 * t + 1.0000000000e+00f;
#endif

        sfpi::dst_reg[0] = n * x * sfpu_reciprocal<APPROXIMATION_MODE>(d2);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void i1_init() {
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
