// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// Pade [5,4] approximation: tanh(t) = t*(945 + 105*t^2 + t^4)/(945 + 420*t^2 + 15*t^4)
// Exact through O(t^11). Accurate to < 1 ULP for |t| < 3.0 in bfloat16.
//
// Reformulated: softcap(x) = x * (945 + 105*u^2 + u^4) / (945 + 420*u^2 + 15*u^4)
// where u = x/cap, and we pre-compute u^2 = x^2 * (1/cap)^2.
//
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_softcap_(
    const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
    sfpi::vFloat cap = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(static_cast<int32_t>(param0)));

    // 1/cap via magic + NR
    sfpi::vInt mg = sfpi::vInt(0x7EF127EA);
    sfpi::vFloat rc = sfpi::reinterpret<sfpi::vFloat>(mg - sfpi::reinterpret<sfpi::vInt>(cap));
    rc = rc * (2.0f - cap * rc);
    rc = rc * (2.0f - cap * rc);

    // (1/cap)^2
    sfpi::vFloat rc2 = rc * rc;

#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // u^2 = x^2 / cap^2
        sfpi::vFloat u2 = x * x * rc2;
        sfpi::vFloat u4 = u2 * u2;

        // Numerator: x * (945 + 105*u^2 + u^4)
        sfpi::vFloat num = x * (945.0f + 105.0f * u2 + u4);

        // Denominator: 945 + 420*u^2 + 15*u^4
        sfpi::vFloat den = 945.0f + 420.0f * u2 + 15.0f * u4;

        // 1/den via magic + NR
        sfpi::vFloat rd = sfpi::reinterpret<sfpi::vFloat>(mg - sfpi::reinterpret<sfpi::vInt>(den));
        rd = rd * (2.0f - den * rd);
        rd = rd * (2.0f - den * rd);

        sfpi::vFloat y = num * rd;

        // Clamp |y| to cap
        v_if(sfpi::abs(y) > cap) {
            y = sfpi::setsgn(cap, x);
        }
        v_endif;

        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
