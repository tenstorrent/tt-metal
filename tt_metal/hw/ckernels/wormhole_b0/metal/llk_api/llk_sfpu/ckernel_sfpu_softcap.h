// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Simplified float-to-int for known positive inputs in the exp_21f range.
sfpi_inline sfpi::vInt float_to_int32_pos_simple_(sfpi::vFloat in) {
    sfpi::vInt exp = exexp(in);
    sfpi::vInt man = exman8(in);
    sfpi::vInt shift = exp - 23;
    return sfpi::reinterpret<sfpi::vInt>(shft(sfpi::reinterpret<sfpi::vUInt>(man), shift));
}

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to [-127, reasonable] before calling.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f_softcap(sfpi::vFloat z) {
    z = sfpi::addexp(z, 23);

    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = float_to_int32_pos_simple_(z + bias);

    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));

    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);

    d2 = d1 * d2;
    sfpi::vInt frac_int = float_to_int32_pos_simple_(d2 * d3);

    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// softcap(x, cap) = cap * tanh(x / cap)
//
// Flat control flow: compute both regimes unconditionally, select via v_if.
// This avoids deeply nested v_if blocks that trigger RISC-V GCC ICE.
//
// vConstFloatPrgm0 = 1/cap (reciprocal)
// vConstFloatPrgm1 = cap
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap() {
    constexpr float tc1 = -0.33333333f;  // -1/3
    constexpr float tc2 = 0.13333333f;   //  2/15
    constexpr float tc3 = -0.05396825f;  // -17/315
    constexpr float tc4 = 0.02186949f;   //  62/2835
    constexpr float neg2_log2e = -2.8853900817779268f;  // -2 * log2(e)

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // u = x / cap = x * (1/cap)
        sfpi::vFloat u = x * sfpi::vConstFloatPrgm0;
        sfpi::vFloat abs_u = sfpi::setsgn(u, 0);

        // Regime 1: degree-9 Taylor polynomial (computed unconditionally)
        sfpi::vFloat u_sq = u * u;
        sfpi::vFloat tanh_u = u * (1.0f + u_sq * (tc1 + u_sq * (tc2 + u_sq * (tc3 + u_sq * tc4))));

        // Regime 2: exp-based (also computed unconditionally, with clamping for safety)
        // z_neg = -2|u| * log2(e), clamped to >= -127 to avoid exp underflow
        sfpi::vFloat z_neg = abs_u * neg2_log2e;
        // Clamp: use max(-127, z_neg) without v_if
        // Since z_neg is always <= 0 (abs_u >= 0, neg2_log2e < 0), and we clamp to -127:
        v_if(z_neg < -127.0f) { z_neg = -127.0f; }
        v_endif;

        sfpi::vFloat f = exp_21f_softcap<APPROXIMATION_MODE>(z_neg);

        // tanh(|u|) = 1 + 2f*(-1 + f*(1 + f*(-1 + f)))
        sfpi::vFloat h = f - 1.0f;
        h = f * h + 1.0f;
        h = f * h - 1.0f;
        sfpi::vFloat tanh_exp = 1.0f + 2.0f * f * h;

        // Apply original sign: tanh is odd, so tanh(-u) = -tanh(u)
        v_if(u < 0.0f) { tanh_exp = -tanh_exp; }
        v_endif;

        // Select regime: use Taylor for |u| < 1.0, exp-based for |u| >= 1.0
        v_if(abs_u >= 1.0f) { tanh_u = tanh_exp; }
        v_endif;

        // result = cap * tanh(u)
        sfpi::dst_reg[0] = sfpi::vConstFloatPrgm1 * tanh_u;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init(float cap) {
    sfpi::vConstFloatPrgm0 = 1.0f / cap;
    sfpi::vConstFloatPrgm1 = cap;
}

}  // namespace ckernel::sfpu
