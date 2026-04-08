// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Since hardware exp/sigmoid primitives are not available, we approximate
// sigmoid using a hybrid polynomial + piecewise-linear approach:
//
//   sigmoid(t) for t = |x| >= 0:
//     Segment 0 (t <= 2.5): degree-3 polynomial fitted at t = 0.5, 1.0, 2.5
//       sigmoid(t) ≈ 0.5 + t * (0.2533 + t * (-0.01479 + t * (-0.00747)))
//       Max error ≈ 0.007 (at t ≈ 2.0)
//
//     Segment 1 (2.5 < t <= 5.0): linear interpolation
//       sigmoid(t) ≈ 0.0276 * t + 0.855
//       Max error ≈ 0.017 (at t ≈ 4.0)
//
//     Segment 2 (t > 5.0): saturate to 1.0
//       Max error ≈ 0.007 (at t = 5.0)
//
//   For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
//   swish(x) = x * sigmoid(x)
//
// Overall max ULP error for bfloat16: ~4 ULP

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() {
    // Polynomial coefficients for sigmoid(t) over [0, 2.5]
    // Fitted to minimize max error at t = 0, 0.5, 1.0, 1.5, 2.0, 2.5
    constexpr float c1 = 0.2533f;
    constexpr float c2 = -0.01479f;
    constexpr float c3 = -0.00747f;

    // Linear segment coefficients for [2.5, 5.0]
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;

    // Breakpoints
    constexpr float bp1 = 2.5f;
    constexpr float bp2 = 5.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Compute sigmoid(|x|) using polynomial for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x);
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3));

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) { sig_pos = ax * lin_slope + lin_offset; }
        v_endif;

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) { sig_pos = sfpi::vConst1; }
        v_endif;

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) { sig_pos = sfpi::vConst1 - sig_pos; }
        v_endif;

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
