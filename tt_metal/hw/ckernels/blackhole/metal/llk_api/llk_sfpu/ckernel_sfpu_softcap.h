// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Branchless float-to-int32 for positive values with exponent >= 0.
// Used only within the Moroz exp_21f algorithm where inputs are always
// in a known positive range, so we skip the negative/overflow checks
// of _float_to_int32_positive_ to reduce register pressure.
sfpi_inline sfpi::vInt softcap_ftoi_pos_(sfpi::vFloat in) {
    sfpi::vInt exp = exexp(in);
    sfpi::vInt man = exman8(in);
    sfpi::vInt shift = exp - 23;
    return sfpi::reinterpret<sfpi::vInt>(shft(sfpi::reinterpret<sfpi::vUInt>(man), shift));
}

// Local exp_21f helper: compute 2^z using Moroz et al. 2022 algorithm.
// Input z should be clamped to >= -127 before calling.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat softcap_exp_21f_(sfpi::vFloat z) {
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23);

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = softcap_ftoi_pos_(z + bias);

    // Step 3: Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);

    d2 = d1 * d2;
    sfpi::vInt frac_int = softcap_ftoi_pos_(d2 * d3);

    // Step 5: Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// softcap(x, cap) = cap * tanh(x / cap)
//
// Computes tanh(u) where u = x / cap using two regimes:
//
// 1. Small |u| (< 1.0): Degree-5 Taylor series
//    tanh(u) = u * (1 + u^2 * (-1/3 + u^2 * (2/15)))
//
// 2. Moderate/large |u| (>= 1.0): Exponential formula
//    Let e = exp(-2|u|). Then:
//    tanh(|u|) = 1 - 2e + 2e^2
//
// Register pressure is minimized by using degree-5 Taylor (not 7)
// and 2-term geometric series (not 3).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t param0) {
    // Decode cap parameter from bit-packed uint32
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = param0;
    const float cap = conv.f;
    const float inv_cap = 1.0f / cap;

    constexpr float neg_2_log2e = -2.0f * 1.4426950408889634f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat u = x * inv_cap;
        sfpi::vFloat abs_u = sfpi::setsgn(u, 0);

        // --- Exp-based regime (computed for all lanes) ---
        sfpi::vFloat z = abs_u * neg_2_log2e;
        v_if(z < -127.0f) { z = -127.0f; }
        v_endif;

        sfpi::vFloat e = softcap_exp_21f_<APPROXIMATION_MODE>(z);

        // tanh(|u|) = 1 - 2e + 2e^2
        sfpi::vFloat result = sfpi::vConst1 - e * 2.0f + e * e * 2.0f;

        // Apply sign: tanh(-u) = -tanh(u)
        v_if(x < 0.0f) { result = -result; }
        v_endif;

        // --- Taylor override for small |u| ---
        // tanh(u) ≈ u * (1 + u^2 * (-1/3 + u^2 * 2/15))
        v_if(abs_u < 1.0f) {
            sfpi::vFloat u2 = u * u;
            result = u * (sfpi::vConst1 + u2 * (-0.33333333f + u2 * 0.13333333f));
        }
        v_endif;

        sfpi::dst_reg[0] = result * cap;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {
    // No programmable constants needed — all coefficients are local constexpr
}

}  // namespace ckernel::sfpu
