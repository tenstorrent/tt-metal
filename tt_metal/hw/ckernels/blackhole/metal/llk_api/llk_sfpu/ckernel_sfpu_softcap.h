// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// Uses tanh(u) = 2*sigmoid(2u) - 1 with polynomial sigmoid.
// Key insight: avoid the 0.5 + tiny -> 0.5 cancellation by computing
// sigmoid_offset = sigmoid(t) - 0.5 directly, then tanh = 2*offset.
//
// sigmoid(t) = 0.5 + t*(c1 + t*(c2 + t*c3)) for t in [0, 2.5]
// offset(t) = t*(c1 + t*(c2 + t*c3))
// tanh(u) = 2*offset(2u)
//
// This preserves precision for small t because we never add to 0.5.
//
// For large t (> 2.5): sigmoid -> 1, so offset -> 0.5, tanh -> 1.
// Linear segment: offset(t) = t*0.0276 + 0.355 for 2.5 < t <= 5
// Saturate: offset = 0.5 for t > 5

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t param0, uint32_t param1) {
    sfpi::vFloat cap = sfpi::s2vFloat16b(param0);
    sfpi::vFloat recip_cap = sfpi::s2vFloat16b(param1);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat ax = sfpi::abs(x);

        // at = 2*|x|/cap
        sfpi::vFloat at = (ax + ax) * recip_cap;

        // offset(at) = sigmoid(at) - 0.5 = at*(c1 + at*(c2 + at*c3))
        // This avoids the 0.5 + tiny cancellation
        sfpi::vFloat off = at * (0.2533f + at * (-0.01479f + at * (-0.00747f)));

        // Linear segment for at > 2.5: sigmoid = at*0.0276 + 0.855
        // offset = at*0.0276 + 0.855 - 0.5 = at*0.0276 + 0.355
        v_if(at > 2.5f) { off = at * 0.0276f + 0.355f; }
        v_endif;

        // Saturate for at > 5: sigmoid = 1, offset = 0.5
        v_if(at > 5.0f) { off = 0.5f; }
        v_endif;

        // tanh(|u|) = 2 * offset(2|u|)
        sfpi::vFloat th = off + off;

        // result = sign(x) * cap * tanh(|u|)
        sfpi::vFloat r = cap * th;
        v_if(x < 0.0f) { r = -r; }
        v_endif;

        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
