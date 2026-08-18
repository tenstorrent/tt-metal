// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Shared piecewise rational P(x)/Q(x) evaluator for LUT-based SFPU activations.
 *
 * The numerator and denominator Horner chains are interleaved so independent
 * SFPMADs can hide pipeline latency. Odd-numerator/even-denominator functions
 * can instead use the x^2 parity evaluator to halve the multiply-add count.
 * Blackhole's ported activation kernels also require the general Horner path;
 * Quasar previously exposed only the parity-specialized evaluator.
 */

#include <array>
#include <cstdint>

#include "ckernel_sfpu_recip.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <std::uint32_t NUM_DEGREE, std::uint32_t DEN_DEGREE>
sfpi_inline void piecewise_rational_eval_numer_denom(
    const float* num_coeffs,
    const float* den_coeffs,
    sfpi::vFloat x,
    sfpi::vFloat& out_numer,
    sfpi::vFloat& out_denom) {
    constexpr std::uint32_t MIN_DEG = (NUM_DEGREE < DEN_DEGREE) ? NUM_DEGREE : DEN_DEGREE;

    sfpi::vFloat number = num_coeffs[NUM_DEGREE];
    sfpi::vFloat denom = den_coeffs[DEN_DEGREE];

    if constexpr (NUM_DEGREE > DEN_DEGREE) {
#pragma GCC unroll 64
        for (int i = NUM_DEGREE - 1; i >= static_cast<int>(DEN_DEGREE); i--) {
            number = number * x + num_coeffs[i];
        }
    } else if constexpr (DEN_DEGREE > NUM_DEGREE) {
#pragma GCC unroll 64
        for (int i = DEN_DEGREE - 1; i >= static_cast<int>(NUM_DEGREE); i--) {
            denom = denom * x + den_coeffs[i];
        }
    }

#pragma GCC unroll 64
    for (int i = MIN_DEG - 1; i >= 0; i--) {
        number = number * x + num_coeffs[i];
        denom = denom * x + den_coeffs[i];
    }

    out_numer = number;
    out_denom = denom;
}

// Coefficient arrays are indexed by power, so the unused parity's entries are zero and get skipped
// via NUM_TOP / DEN_TOP below.

template <std::uint32_t NUM_DEGREE, std::uint32_t DEN_DEGREE>
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

    sfpi::vFloat number = num_coeffs[NUM_TOP];
    sfpi::vFloat denom = den_coeffs[DEN_TOP];

    if constexpr (NUM_STEPS > DEN_STEPS) {
#pragma GCC unroll 64
        for (int k = 0; k < NUM_STEPS - DEN_STEPS; k++) {
            number = number * x2 + num_coeffs[NUM_TOP - 2 * (k + 1)];
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
        number = number * x2 + num_coeffs[NUM_POS - 2 * k];
        denom = denom * x2 + den_coeffs[DEN_POS - 2 * k];
    }

    out_numer = number * x;  // odd parity: P(x) = x * Horner_result
    out_denom = denom;
}

template <std::uint32_t NUM_DEGREE, std::uint32_t DEN_DEGREE, bool USE_PARITY = false>
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

template <
    std::uint32_t SEG,
    std::uint32_t NUM_DEGREE,
    std::uint32_t DEN_DEGREE,
    std::uint32_t NUM_SEGMENTS,
    std::uint32_t LUT_SIZE,
    bool USE_PARITY = false>
sfpi_inline void piecewise_rational_unroll_segment(
    const std::array<float, LUT_SIZE>& lut,
    sfpi::vFloat x,
    sfpi::vFloat& number,
    sfpi::vFloat& denom,
    sfpi::vFloat x2 = 0.0f) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr std::uint32_t NUM_COEFFS = NUM_DEGREE + 1;
        constexpr std::uint32_t COEFFS_PER_SEGMENT = NUM_COEFFS + DEN_DEGREE + 1;
        constexpr std::uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;
        v_if(x >= lut[SEG]) {
            piecewise_rational_dispatch_numer_denom<NUM_DEGREE, DEN_DEGREE, USE_PARITY>(
                &lut[COEFF_OFFSET + SEG * COEFFS_PER_SEGMENT],
                &lut[COEFF_OFFSET + SEG * COEFFS_PER_SEGMENT + NUM_COEFFS],
                x,
                number,
                denom,
                x2);
        }
        v_endif;
        piecewise_rational_unroll_segment<SEG + 1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE, USE_PARITY>(
            lut, x, number, denom, x2);
    }
}

template <
    std::uint32_t NUM_DEGREE,
    std::uint32_t DEN_DEGREE,
    std::uint32_t NUM_SEGMENTS,
    std::uint32_t LUT_SIZE,
    bool USE_PARITY = false,
    bool APPROX_RECIP = false>
sfpi_inline sfpi::vFloat piecewise_rational_eval(const std::array<float, LUT_SIZE>& lut, sfpi::vFloat x) {
    constexpr std::uint32_t NUM_COEFFS = NUM_DEGREE + 1;
    constexpr std::uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    sfpi::vFloat x2;
    if constexpr (USE_PARITY) {
        x2 = x * x;
    }

    sfpi::vFloat number = 0.0f;
    sfpi::vFloat denom = 0.0f;
    piecewise_rational_dispatch_numer_denom<NUM_DEGREE, DEN_DEGREE, USE_PARITY>(
        &lut[COEFF_OFFSET], &lut[COEFF_OFFSET + NUM_COEFFS], x, number, denom, x2);

    if constexpr (NUM_SEGMENTS > 1) {
        piecewise_rational_unroll_segment<1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE, USE_PARITY>(
            lut, x, number, denom, x2);
    }

    return number * _sfpu_reciprocal_<APPROX_RECIP ? 0 : 2>(denom);
}

}  // namespace ckernel::sfpu
