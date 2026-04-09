// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// tanh is approximated using a piecewise approach (similar to swish sigmoid):
//
// Region 1 (|u| <= 1.0): degree-5 odd polynomial via Horner form
//   tanh(u) ≈ u * (1 + u² * (-0.3253 + u² * 0.0869))
//   Fitted through tanh(0) = 0, tanh(0.5) = 0.4621, tanh(1.0) = 0.7616
//   Max error: ~0.3% relative (~0.8 ULP bfloat16)
//
// Region 2 (1.0 < |u| <= 2.0): quadratic in u
//   tanh(u) ≈ 0.2208 + 0.710 * u - 0.1692 * u²
//   Fitted through tanh(1.0), tanh(1.5), tanh(2.0)
//   Max error: ~0.5% relative (~1.3 ULP bfloat16)
//
// Region 3 (2.0 < |u| <= 3.0): quadratic in u
//   tanh(u) ≈ 0.7324 + 0.1722 * u - 0.0282 * u²
//   Fitted through tanh(2.0), tanh(2.5), tanh(3.0)
//   Max error: ~0.2% relative (~0.5 ULP bfloat16)
//
// Region 4 (|u| > 3.0): tanh(u) = 1.0
//   tanh(3.0) = 0.9951, rounds to 0.996 in bf16; returning 1.0 is ~1 ULP error
//
// The parameter cap is passed as a uint32_t (bit-cast from float).
// In init, we precompute recip_cap = 1/cap and store cap and recip_cap
// in programmable constant registers.

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t param0) {
    const float cap = std::bit_cast<float>(param0);
    const float recip_cap = 1.0f / cap;

    // Region 1 polynomial coefficients (tanh(u) = u * (p0 + u² * (p1 + u² * p2)))
    constexpr float p0 = 1.0f;
    constexpr float p1 = -0.3253f;
    constexpr float p2 = 0.0869f;

    // Region 2 quadratic coefficients: tanh(u) = r2a + r2b * u + r2c * u²
    constexpr float r2a = 0.2208f;
    constexpr float r2b = 0.710f;
    constexpr float r2c = -0.1692f;

    // Region 3 quadratic coefficients: tanh(u) = r3a + r3b * u + r3c * u²
    constexpr float r3a = 0.7324f;
    constexpr float r3b = 0.1722f;
    constexpr float r3c = -0.0282f;

    // Breakpoints
    constexpr float bp1 = 1.0f;
    constexpr float bp2 = 2.0f;
    constexpr float bp3 = 3.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // u = x / cap
        sfpi::vFloat u = x * recip_cap;
        sfpi::vFloat au = sfpi::abs(u);

        // Region 1: |u| <= 1.0
        // tanh(au) = au * (p0 + au² * (p1 + au² * p2))
        sfpi::vFloat au_sq = au * au;
        sfpi::vFloat poly = au_sq * p2 + p1;
        poly = poly * au_sq + p0;
        sfpi::vFloat tanh_val = au * poly;

        // Region 2: 1.0 < |u| <= 2.0
        v_if(au > bp1) { tanh_val = r2a + au * (r2b + au * r2c); }
        v_endif;

        // Region 3: 2.0 < |u| <= 3.0
        v_if(au > bp2) { tanh_val = r3a + au * (r3b + au * r3c); }
        v_endif;

        // Region 4: |u| > 3.0 -> tanh = 1.0
        v_if(au > bp3) { tanh_val = sfpi::vConst1; }
        v_endif;

        // result = cap * tanh_val * sign(u)
        // Since u = x * recip_cap, sign(u) = sign(x), so:
        // result = cap * tanh_val for x >= 0, -cap * tanh_val for x < 0
        sfpi::vFloat result = tanh_val * cap;

        v_if(x < 0.0f) { result = -result; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init(uint32_t param0) {
    // No programmable constants needed - all coefficients are compile-time
    // constants loaded via SFPLOADI in the main loop.
    // The cap parameter is passed directly to calculate_softcap.
}

}  // namespace ckernel::sfpu
