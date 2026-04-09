// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel::sfpu {

// Helper: compute 2^z using fast exp algorithm.
// Uses IEEE 754 bit manipulation: interprets (z + 127) * 2^23 as float bits.
// Input z must be clamped to [-126, 126] to avoid overflow.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat softcap_exp2(sfpi::vFloat z) {
    // Scale z by 2^23
    z = sfpi::addexp(z, 23);
    // Add 127 * 2^23 (= 0x3F800000 as integer = 1065353216.0 as float)
    // vFloat(int) constructs from raw IEEE bits, so vFloat(0x3f800000) = 1.0
    // We need to add the VALUE 127*2^23 = 1065353216.0
    z = z + 1065353216.0f;
    // The bit pattern of z now approximates 2^z_orig
    // Reinterpret the float value's bits as the result
    // For values >= 2^23, the float is exact, so reinterpret IS the bit pattern
    return z;
}

// softcap(x, cap) = cap * tanh(x / cap)
//
// Piecewise polynomial approximation for tanh(u) where u = x/cap:
//
// Region 1 — small |u| < 0.9:
//   Degree-7 Taylor:  tanh(u) ≈ u·(1 + u²·(-1/3 + u²·(2/15 + u²·(-17/315))))
//   Excellent accuracy (<1 ULP bfloat16) for small arguments.
//
// Region 2 — medium 0.9 ≤ |u| < 3.0:
//   Degree-3 polynomial fitted at Chebyshev nodes:
//   tanh(t) ≈ 0.09285 + 0.99375·t - 0.372·t² + 0.047·t³
//   Max relative error ~0.5% (~1 ULP bfloat16).
//
// Region 3 — large |u| ≥ 3.0:
//   Saturate tanh to 1.0.
//
// Final: result = sign(x) · cap · tanh(|u|)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(float cap) {
    const float inv_cap = 1.0f / cap;

    // Taylor coefficients for tanh: u*(1 + u²*(c3 + u²*(c5 + u²*c7)))
    constexpr float c3 = -0.33333333f;  // -1/3
    constexpr float c5 = 0.13333333f;   //  2/15
    constexpr float c7 = -0.05396825f;  // -17/315

    // Degree-3 polynomial coefficients for tanh on [0.9, 3.0]
    // Fitted through (1.0, 0.7616), (1.5, 0.9051), (2.5, 0.9866), (3.0, 0.9951)
    constexpr float p0 = 0.09285f;
    constexpr float p1 = 0.99375f;
    constexpr float p2 = -0.372f;
    constexpr float p3 = 0.047f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // u = x / cap
        sfpi::vFloat u = x * inv_cap;
        sfpi::vFloat abs_u = sfpi::setsgn(u, 0);

        // Default: degree-3 polynomial for medium |u| (computed for all lanes)
        // tanh(t) ≈ p0 + p1*t + p2*t² + p3*t³ = p0 + t*(p1 + t*(p2 + t*p3))
        sfpi::vFloat tanh_abs = p0 + abs_u * (p1 + abs_u * (p2 + abs_u * p3));

        // --- Taylor path for small |u| < 0.9 (overrides polynomial result) ---
        v_if(abs_u < 0.9f) {
            sfpi::vFloat u2 = u * u;
            // Horner: u * (1 + u²*(-1/3 + u²*(2/15 + u²*(-17/315))))
            sfpi::vFloat taylor = u * (sfpi::vConst1 + u2 * (c3 + u2 * (c5 + u2 * c7)));
            tanh_abs = sfpi::setsgn(taylor, 0);
        }
        v_endif;

        // --- Saturation for large |u| ≥ 3.0 ---
        v_if(abs_u >= 3.0f) { tanh_abs = sfpi::vConst1; }
        v_endif;

        // result = sign(x) * cap * tanh(|u|)
        sfpi::vFloat result = sfpi::setsgn(cap * tanh_abs, x);

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {
    // No programmable constants needed
}

}  // namespace ckernel::sfpu
