// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Rational P(x)/Q(x) evaluator for SFPU activations. Carries over the parity evaluator from the
 * Blackhole header of the same name; the other variants there have no Quasar users yet.
 *
 * For an odd numerator and even denominator (erf, atanh, erfinv) both polynomials are evaluated in
 * the x^2 basis, halving the multiply-add count. The two Horner chains are independent, so their
 * SFPMADs interleave and hide pipeline latency.
 */

#include "sfpi.h"

namespace ckernel::sfpu {

// Coefficient arrays are indexed by power, so the unused parity's entries are zero and get skipped
// via NUM_TOP / DEN_TOP below.

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
sfpi_inline void piecewise_rational_eval_parity_numer_denom(
    const float* num_coeffs,
    const float* den_coeffs,
    sfpi::vFloat x,
    sfpi::vFloat x2,
    sfpi::vFloat& out_numer,
    sfpi::vFloat& out_denom) {
    // NUM_TOP/DEN_TOP: highest odd/even index used in x²-Horner.
    // If NUM_DEGREE is even, the leading (even-index) coeff must be zero — we skip it.
    constexpr int NUM_TOP = (NUM_DEGREE % 2 == 1) ? NUM_DEGREE : NUM_DEGREE - 1;
    constexpr int DEN_TOP = (DEN_DEGREE % 2 == 0) ? DEN_DEGREE : DEN_DEGREE - 1;
    constexpr int NUM_STEPS = (NUM_TOP - 1) / 2;
    constexpr int DEN_STEPS = DEN_TOP / 2;

    sfpi::vFloat numer = num_coeffs[NUM_TOP];
    sfpi::vFloat denom = den_coeffs[DEN_TOP];

    if constexpr (NUM_STEPS > DEN_STEPS) {
#pragma GCC unroll 64
        for (int k = 0; k < NUM_STEPS - DEN_STEPS; k++) {
            numer = numer * x2 + num_coeffs[NUM_TOP - 2 * (k + 1)];
        }
    } else if constexpr (DEN_STEPS > NUM_STEPS) {
#pragma GCC unroll 64
        for (int k = 0; k < DEN_STEPS - NUM_STEPS; k++) {
            denom = denom * x2 + den_coeffs[DEN_TOP - 2 * (k + 1)];
        }
    }

    constexpr int MIN_STEPS = (NUM_STEPS < DEN_STEPS) ? NUM_STEPS : DEN_STEPS;
    constexpr int NUM_POS = NUM_TOP - 2 * ((NUM_STEPS > DEN_STEPS) ? (NUM_STEPS - DEN_STEPS) : 0);
    constexpr int DEN_POS = DEN_TOP - 2 * ((DEN_STEPS > NUM_STEPS) ? (DEN_STEPS - NUM_STEPS) : 0);

#pragma GCC unroll 64
    for (int k = 1; k <= MIN_STEPS; k++) {
        numer = numer * x2 + num_coeffs[NUM_POS - 2 * k];
        denom = denom * x2 + den_coeffs[DEN_POS - 2 * k];
    }

    out_numer = numer * x;  // odd parity: P(x) = x * Horner_result
    out_denom = denom;
}

}  // namespace ckernel::sfpu
