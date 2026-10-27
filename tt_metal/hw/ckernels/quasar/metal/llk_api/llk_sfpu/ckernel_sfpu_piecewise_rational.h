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
 *
 * The interleaved (non-parity) evaluator, the segment unroller and the piecewise_rational_eval entry
 * point are the Blackhole ones, for the ported erf / erfc / digamma kernels; only the reciprocal is
 * spelled the Quasar way. The Blackhole range-reduction variants still have no Quasar users.
 */

#include <array>

#include "ckernel_sfpu_recip.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// ============================================================================
// Interleaved Horner: evaluate P(x) and Q(x) simultaneously
// Back-to-back SFPMADs on independent chains hide pipeline latency.
// ============================================================================

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
sfpi_inline void piecewise_rational_eval_numer_denom(
    const float* num_coeffs,
    const float* den_coeffs,
    sfpi::vFloat x,
    sfpi::vFloat& out_numer,
    sfpi::vFloat& out_denom) {
    constexpr uint32_t MIN_DEG = (NUM_DEGREE < DEN_DEGREE) ? NUM_DEGREE : DEN_DEGREE;

    sfpi::vFloat numer = num_coeffs[NUM_DEGREE];
    sfpi::vFloat denom = den_coeffs[DEN_DEGREE];

    if constexpr (NUM_DEGREE > DEN_DEGREE) {
#pragma GCC unroll 64
        for (int i = NUM_DEGREE - 1; i >= static_cast<int>(DEN_DEGREE); i--) {
            numer = numer * x + num_coeffs[i];
        }
    } else if constexpr (DEN_DEGREE > NUM_DEGREE) {
#pragma GCC unroll 64
        for (int i = DEN_DEGREE - 1; i >= static_cast<int>(NUM_DEGREE); i--) {
            denom = denom * x + den_coeffs[i];
        }
    }

#pragma GCC unroll 64
    for (int i = MIN_DEG - 1; i >= 0; i--) {
        numer = numer * x + num_coeffs[i];
        denom = denom * x + den_coeffs[i];
    }

    out_numer = numer;
    out_denom = denom;
}

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

// ============================================================================
// Unified numer/denom dispatcher: selects parity or interleaved automatically
// ============================================================================

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, bool USE_PARITY = false>
sfpi_inline void piecewise_rational_dispatch_numer_denom(
    const float* num_coeffs,
    const float* den_coeffs,
    sfpi::vFloat x,
    sfpi::vFloat& out_numer,
    sfpi::vFloat& out_denom,
    sfpi::vFloat x2 = 0.0f) {
    if constexpr (USE_PARITY) {
        piecewise_rational_eval_parity_numer_denom<NUM_DEGREE, DEN_DEGREE>(
            num_coeffs, den_coeffs, x, x2, out_numer, out_denom);
    } else {
        piecewise_rational_eval_numer_denom<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x, out_numer, out_denom);
    }
}

// ============================================================================
// Recursive segment unroller with deferred reciprocal
// ============================================================================

template <
    uint32_t SEG,
    uint32_t NUM_DEGREE,
    uint32_t DEN_DEGREE,
    uint32_t NUM_SEGMENTS,
    uint32_t LUT_SIZE,
    bool USE_PARITY = false>
sfpi_inline void piecewise_rational_unroll_segment(
    const std::array<float, LUT_SIZE>& lut,
    sfpi::vFloat x,
    sfpi::vFloat& numer,
    sfpi::vFloat& denom,
    sfpi::vFloat x2 = 0.0f) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1;
        constexpr uint32_t CPS = NUM_COEFFS + DEN_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
        v_if(x >= lut[SEG]) {
            piecewise_rational_dispatch_numer_denom<NUM_DEGREE, DEN_DEGREE, USE_PARITY>(
                &lut[CO + SEG * CPS], &lut[CO + SEG * CPS + NUM_COEFFS], x, numer, denom, x2);
        }
        v_endif;
        piecewise_rational_unroll_segment<SEG + 1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE, USE_PARITY>(
            lut, x, numer, denom, x2);
    }
}

// ============================================================================
// Public API: evaluate piecewise rational LUT for a single vFloat x
// Automatically dispatches parity when macros are defined.
// USE_PARITY template parameter allows per-call parity control without macros.
// ============================================================================

template <
    uint32_t NUM_DEGREE,
    uint32_t DEN_DEGREE,
    uint32_t NUM_SEGMENTS,
    uint32_t LUT_SIZE,
    bool USE_PARITY = false,
    bool APPROX_RECIP = false>
sfpi_inline sfpi::vFloat piecewise_rational_eval(const std::array<float, LUT_SIZE>& lut, sfpi::vFloat x) {
    constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    // Parity controlled exclusively via template parameter — no macro leakage
    constexpr bool parity_active = USE_PARITY;

    sfpi::vFloat x2;
    if constexpr (parity_active) {
        x2 = x * x;
    }

    sfpi::vFloat numer = 0.0f, denom = 0.0f;

    if constexpr (parity_active) {
        piecewise_rational_eval_parity_numer_denom<NUM_DEGREE, DEN_DEGREE>(
            &lut[COEFF_OFFSET], &lut[COEFF_OFFSET + NUM_COEFFS], x, x2, numer, denom);
    } else {
        piecewise_rational_eval_numer_denom<NUM_DEGREE, DEN_DEGREE>(
            &lut[COEFF_OFFSET], &lut[COEFF_OFFSET + NUM_COEFFS], x, numer, denom);
    }

    // Unroll remaining segments (seg 1..N-1)
    if constexpr (NUM_SEGMENTS > 1) {
        piecewise_rational_unroll_segment<1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE, USE_PARITY>(
            lut, x, numer, denom, x2);
    }

    return numer * _sfpu_reciprocal_<APPROX_RECIP ? 0 : 2>(denom);
}

}  // namespace ckernel::sfpu
