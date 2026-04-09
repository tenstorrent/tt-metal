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
// [3,3] Padé rational approximant for tanh:
//   tanh(u) = u * (u⁴ + 105*u² + 945) / (15*u⁴ + 420*u² + 945)
// Matches Taylor series through order u⁷.
// Result clamped to [-1, 1] to handle overshoot for |u| > ~3.7.
// Division via Newton-Raphson reciprocal (2 iterations, ~24-bit accuracy).
//
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t param_cap, uint32_t param_inv_cap) {
    const sfpi::vFloat cap = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(param_cap));
    const sfpi::vFloat inv_cap = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(param_inv_cap));

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat u = sfpi::dst_reg[0] * inv_cap;
        sfpi::vFloat t = u * u;

        // Numerator: N(t) = t² + 105t + 945
        sfpi::vFloat n = (t + 105.0f) * t + 945.0f;

        // Denominator: D(t) = 15t² + 420t + 945
        sfpi::vFloat dd = (15.0f * t + 420.0f) * t + 945.0f;

        // Newton-Raphson reciprocal of dd (2 iterations)
        sfpi::vFloat r = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(0x7EF311C3) - sfpi::reinterpret<sfpi::vInt>(dd));
        r = r * (2.0f - dd * r);
        r = r * (2.0f - dd * r);

        // tanh(u) = u * N / D, clamped to [-1, 1]
        sfpi::vFloat result = u * n * r;

        // Clamp to [-1, 1] (handles Padé overshoot for |u| > 3.7 and saturation)
        v_if(result > 1.0f) { result = 1.0f; }
        v_endif;
        v_if(result < -1.0f) { result = -1.0f; }
        v_endif;

        sfpi::dst_reg[0] = cap * result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {}

}  // namespace ckernel::sfpu
