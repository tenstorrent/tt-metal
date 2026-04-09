// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Local exp_21f helper: compute 2^z using Moroz et al. 2022 algorithm.
// Input z should be clamped to >= -127 before calling.
// Copied from ckernel_sfpu_sinh.h to avoid cross-include dependencies.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat softcap_exp_21f_(sfpi::vFloat z) {
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23);

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias);

    // Step 3: Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);

    d2 = d1 * d2;
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3);

    // Step 5: Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// softcap(x, cap) = cap * tanh(x / cap)
//
// Computes tanh(u) where u = x / cap using two regimes:
//
// 1. Small |u| (< 1.0): Degree-7 Taylor series
//    tanh(u) = u * (1 + u^2 * (-1/3 + u^2 * (2/15 + u^2 * (-17/315))))
//    This provides <1 ULP accuracy in bfloat16 for |u| < 0.85,
//    and ~2 ULP at |u| = 1.0.
//
// 2. Moderate/large |u| (>= 1.0): Exponential series
//    Let e = exp(-2|u|). Then:
//    tanh(|u|) = 1 - 2e + 2e^2 - 2e^3
//    This is the geometric series expansion of (1-e)/(1+e) truncated
//    at degree 3. For |u| >= 1, e <= 0.135, giving error < 0.2 ULP.
//    For |u| >= 4, e < 1e-5 and tanh rounds to 1.0 in bfloat16.
//
// The result is then multiplied by cap to get softcap(x).
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

    // Taylor degree-7 coefficients for tanh(u) in Horner form on u^2:
    // tanh(u) = u * (1 + u^2 * (c1 + u^2 * (c2 + u^2 * c3)))
    constexpr float c1 = -1.0f / 3.0f;     // -0.33333333
    constexpr float c2 = 2.0f / 15.0f;     //  0.13333333
    constexpr float c3 = -17.0f / 315.0f;  // -0.05396825

    // Threshold: use Taylor below this, exp above
    constexpr float taylor_bound = 1.0f;

    // For exp-based computation: log2(e) for base conversion
    constexpr float neg_2_log2e = -2.0f * 1.4426950408889634f;  // -2 * log2(e)
    const sfpi::vFloat v_low_threshold = -127.0f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat u = x * inv_cap;
        sfpi::vFloat abs_u = sfpi::setsgn(u, 0);

        // --- Exp-based regime (computed for all lanes) ---
        // e = exp(-2|u|) = 2^(-2|u| * log2(e))
        sfpi::vFloat z = abs_u * neg_2_log2e;
        v_if(z < v_low_threshold) { z = v_low_threshold; }
        v_endif;

        sfpi::vFloat e = softcap_exp_21f_<APPROXIMATION_MODE>(z);
        sfpi::vFloat e2 = e * e;

        // tanh(|u|) = 1 - 2e + 2e^2 - 2e^3
        sfpi::vFloat tanh_abs = sfpi::vConst1 - e * 2.0f;
        tanh_abs = tanh_abs + e2 * 2.0f;
        tanh_abs = tanh_abs - e2 * e * 2.0f;

        // Apply sign of u
        sfpi::vFloat tanh_u = tanh_abs;
        v_if(x < 0.0f) { tanh_u = -tanh_abs; }
        v_endif;

        // --- Taylor override for small |u| ---
        // tanh(u) = u * (1 + u^2 * (-1/3 + u^2 * (2/15 + u^2 * (-17/315))))
        v_if(abs_u < taylor_bound) {
            sfpi::vFloat u2 = u * u;
            sfpi::vFloat poly = u2 * c3 + c2;
            poly = poly * u2 + c1;
            poly = poly * u2 + sfpi::vConst1;
            tanh_u = u * poly;
        }
        v_endif;

        // softcap(x) = cap * tanh(x / cap)
        sfpi::dst_reg[0] = tanh_u * cap;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {
    // No programmable constants needed — all coefficients are local constexpr
}

}  // namespace ckernel::sfpu
