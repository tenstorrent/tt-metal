// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// Uses tanh(u) = 2*sigmoid(2u) - 1, with the same sigmoid approximation
// as the swish kernel (piecewise polynomial for positive arguments).
//
// For sigmoid(s) where s = 2*|u| >= 0:
//   Segment 0 (s <= 2.5): cubic polynomial
//     sigmoid(s) = 0.5 + s*(0.2533 + s*(-0.01479 + s*(-0.00747)))
//   Segment 1 (2.5 < s <= 5.0): linear
//     sigmoid(s) = 0.0276*s + 0.855
//   Segment 2 (s > 5.0): 1.0
//
// tanh(|u|) = 2*sigmoid(2|u|) - 1
// softcap(x, cap) = sign(x) * cap * tanh(|x/cap|)
//
// For small |u| (<= 0.25), use Taylor series:
//   tanh(u) ~ u*(1 - u^2/3)
//   softcap = x*(1 - u^2/3)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t param0) {
    union {
        uint32_t u;
        float f;
    } cap_conv;
    cap_conv.u = param0;
    const float cap = cap_conv.f;
    const float inv_cap = 1.0f / cap;

    // Sigmoid polynomial coefficients (from swish kernel)
    constexpr float pc1 = 0.2533f;
    constexpr float pc2 = -0.01479f;
    constexpr float pc3 = -0.00747f;

    // Linear segment coefficients
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;

    // Breakpoints
    constexpr float bp1 = 2.5f;
    constexpr float bp2 = 5.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // u = x/cap, au = |u|, s = 2*|u|
        sfpi::vFloat u = x * inv_cap;
        sfpi::vFloat au = sfpi::abs(u);
        sfpi::vFloat s = au + au;

        // Compute sigmoid(s) for s >= 0, using same approach as swish
        sfpi::vFloat sig = 0.5f + s * (pc1 + s * (pc2 + s * pc3));

        // Override with linear for s > 2.5
        v_if(s > bp1) { sig = s * lin_slope + lin_offset; }
        v_endif;

        // Saturate for s > 5.0
        v_if(s > bp2) { sig = sfpi::vConst1; }
        v_endif;

        // tanh(|u|) = 2*sigmoid(s) - 1
        sfpi::vFloat tanh_val = sig + sig - sfpi::vConst1;

        // softcap = cap * tanh(u) = sign(x) * cap * tanh(|u|)
        sfpi::vFloat result = cap * tanh_val;
        v_if(x < 0.0f) { result = -result; }
        v_endif;

        // Small |u| override: Taylor for better precision near zero
        // tanh(u) ≈ u - u³/3, so softcap = cap*u*(1-u²/3) = x*(1-u²/3)
        v_if(au < 0.25f) {
            sfpi::vFloat usq = u * u;
            result = x * (sfpi::vConst1 - usq * 0.3333333f);
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
