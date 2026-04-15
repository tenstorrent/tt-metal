// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// Three regimes:
//   Tiny |u|  (< 4e-4):   softcap ≈ x  (identity, avoids rounding in cap*(x/cap))
//   Small |u| (< 0.5):    Taylor degree-13 for tanh, result = x * poly(u²)
//   Large |u| (≥ 0.5):    exp-based tanh = 1 - 2/(1+exp(2|u|))
//   Saturated |u| (≥ 9):  tanh = ±1, softcap = ±cap

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(float cap) {
    const float inv_cap = 1.0f / cap;

    // Taylor coefficients: tanh(u)/u = 1 + T1*u² + T2*u⁴ + ... + T6*u¹²
    constexpr float T1 = -0.33333333333333333f;
    constexpr float T2 = 0.13333333333333333f;
    constexpr float T3 = -0.053968253968253968f;
    constexpr float T4 = 0.021869488536155203f;
    constexpr float T5 = -0.0088632355299021976f;
    constexpr float T6 = 0.0035921280365724800f;

    // Exp constants
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2_HI = 0.6931457519531250f;
    constexpr float LN2_LO = 1.4286067653502116e-06f;
    constexpr float ROUND_MAGIC = 12582912.0f;
    constexpr int ROUND_MAGIC_INT = 0x4B400000;

    // Exp polynomial: degree 7
    constexpr float E7 = 0.0001984127f;
    constexpr float E6 = 0.0013888889f;
    constexpr float E5 = 0.0083333333f;
    constexpr float E4 = 0.0416666667f;
    constexpr float E3 = 0.1666666667f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat u = x * inv_cap;
        sfpi::vFloat au = sfpi::abs(u);

        // ---- Default path: exp-based tanh for |u| >= 0.5 ----
        sfpi::vFloat y = au + au;

        sfpi::vFloat n_biased = y * LOG2E + ROUND_MAGIC;
        sfpi::vFloat n_float = n_biased - ROUND_MAGIC;
        sfpi::vFloat r = y - n_float * LN2_HI;
        r = r - n_float * LN2_LO;

        // exp(r) via degree-7 Horner
        sfpi::vFloat p = r * E7 + E6;
        p = p * r + E5;
        p = p * r + E4;
        p = p * r + E3;
        p = p * r + 0.5f;
        p = p * r + 1.0f;
        p = p * r + 1.0f;

        sfpi::vInt n_int = sfpi::reinterpret<sfpi::vInt>(n_biased) - ROUND_MAGIC_INT;
        sfpi::vInt old_exp = sfpi::exexp_nodebias(p);
        sfpi::vFloat exp_y = sfpi::setexp(p, sfpi::reinterpret<sfpi::vUInt>(old_exp + n_int));

        // tanh = 1 - 2/(exp_y + 1) via Newton-Raphson reciprocal (4 iterations)
        sfpi::vFloat den = exp_y + 1.0f;
        sfpi::vInt recip_int = sfpi::vInt(0x7F000000) - sfpi::reinterpret<sfpi::vInt>(den);
        sfpi::vFloat recip = sfpi::reinterpret<sfpi::vFloat>(recip_int);
        recip = recip * (2.0f - den * recip);
        recip = recip * (2.0f - den * recip);
        recip = recip * (2.0f - den * recip);
        recip = recip * (2.0f - den * recip);

        sfpi::vFloat two_r = recip + recip;
        sfpi::vFloat result = (1.0f - two_r) * cap;
        // Apply sign for the exp path
        v_if(u < 0.0f) { result = -result; }
        v_endif;

        // ---- Taylor path: |u| < 0.5 ----
        // softcap = cap * u * poly(u²) = x * poly(u²)
        // Using x * poly avoids the rounding error from cap * (x/cap)
        v_if(au < 0.5f) {
            sfpi::vFloat s = u * u;
            sfpi::vFloat poly = s * T6 + T5;
            poly = poly * s + T4;
            poly = poly * s + T3;
            poly = poly * s + T2;
            poly = poly * s + T1;
            poly = poly * s + 1.0f;
            // softcap = cap * tanh(u) = cap * u * poly = x * poly
            result = x * poly;
        }
        v_endif;

        // ---- Tiny path: |u| < 4e-4 ----
        // tanh(u) ≈ u with < 0.5 ULP error, softcap ≈ x exactly
        v_if(au < 4.0e-4f) { result = x; }
        v_endif;

        // ---- Saturation: |u| >= 9 ----
        v_if(au > 9.0f) {
            result = cap;
            v_if(u < 0.0f) { result = -result; }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
