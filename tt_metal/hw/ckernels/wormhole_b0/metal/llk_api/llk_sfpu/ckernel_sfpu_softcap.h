// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// Uses Padé [7,6] rational approximation for tanh:
//   tanh(u) = u * P(u²) / Q(u²)
// where:
//   P(t) = 135135 + 17325*t + 378*t² + t³
//   Q(t) = 135135 + 62370*t + 3150*t² + 28*t³
//
// Since softcap(x) = cap * tanh(x/cap) = x * P(u²) / Q(u²) where u = x/cap,
// the cap factor cancels in the non-saturated regime.
//
// Division 1/Q is computed via IEEE 754 decomposition + Newton-Raphson:
//   1. Extract exponent/mantissa of Q
//   2. Quadratic initial estimate of 1/mantissa
//   3. Reconstruct reciprocal with correct exponent
//   4. Three Newton-Raphson iterations for ~28-bit precision
//
// Segments:
//   |u| < 4.0: Padé [7,6] approximation (degree 7 rational)
//   |u| >= 4.0: saturate to ±cap (tanh(u) ≈ ±1 for |u| > 4)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t param0) {
    // Interpret the packed uint32 parameter as float cap value
    float cap;
    {
        uint32_t tmp = param0;
        __builtin_memcpy(&cap, &tmp, sizeof(float));
    }
    const float inv_cap = 1.0f / cap;
    const float inv_cap_sq = inv_cap * inv_cap;

    // Padé [7,6] coefficients for tanh(u) = u * P(u²) / Q(u²)
    // P(t) = 135135 + 17325*t + 378*t² + t³
    // Q(t) = 135135 + 62370*t + 3150*t² + 28*t³
    constexpr float p0 = 135135.0f;
    constexpr float p1 = 17325.0f;
    constexpr float p2 = 378.0f;
    // p3 = 1.0f (implicit in Horner's)
    constexpr float q0 = 135135.0f;
    constexpr float q1 = 62370.0f;
    constexpr float q2 = 3150.0f;
    constexpr float q3 = 28.0f;

    // Quadratic coefficients for initial reciprocal estimate of 1/m, m in [1,2)
    // Interpolation fit: f(1)=1, f(1.5)=2/3, f(2)=0.5
    constexpr float rc0 = 13.0f / 6.0f;  // 2.166667
    constexpr float rc1 = -1.5f;
    constexpr float rc2 = 1.0f / 3.0f;  // 0.333333

    constexpr float sat_threshold = 4.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // u² = (x/cap)² = x² * inv_cap²
        sfpi::vFloat x_sq = x * x;
        sfpi::vFloat u_sq = x_sq * inv_cap_sq;
        sfpi::vFloat au = sfpi::abs(x) * inv_cap;

        // Horner's evaluation of P(u²) and Q(u²)
        sfpi::vFloat P = p0 + u_sq * (p1 + u_sq * (p2 + u_sq));
        sfpi::vFloat Q = q0 + u_sq * (q1 + u_sq * (q2 + q3 * u_sq));

        // Compute 1/Q using IEEE 754 decomposition + Newton-Raphson
        // Step 1: Extract exponent and mantissa of Q
        sfpi::vInt eQ = sfpi::exexp(Q);          // unbiased exponent of Q
        sfpi::vFloat mQ = sfpi::setexp(Q, 127);  // mantissa normalized to [1, 2)

        // Step 2: Quadratic initial estimate of 1/m for m in [1, 2)
        sfpi::vFloat rm = rc0 + mQ * (rc1 + mQ * rc2);

        // Step 3: Reconstruct 1/Q = rm * 2^(-eQ)
        // rm has exponent erm (from its own float representation)
        sfpi::vInt erm = sfpi::exexp(rm);
        sfpi::vFloat recip_Q = sfpi::setexp(rm, 127 + erm - eQ);

        // Step 4: Newton-Raphson refinement: y = y * (2 - Q * y)
        // Each iteration roughly doubles precision bits
        recip_Q = recip_Q * (2.0f - Q * recip_Q);  // ~12 bits
        recip_Q = recip_Q * (2.0f - Q * recip_Q);  // ~24 bits
        recip_Q = recip_Q * (2.0f - Q * recip_Q);  // ~48 bits (clamped to fp32)

        // softcap(x) = x * P(u²) / Q(u²) = x * P * (1/Q)
        sfpi::vFloat result = x * P * recip_Q;

        // Saturation: for |u| >= 4, tanh(u) ≈ sign(u), so softcap(x) = sign(x) * cap
        v_if(au >= sat_threshold) {
            result = cap;
            v_if(x < 0.0f) { result = -cap; }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {
    // No programmable constants needed — all coefficients are compile-time constants
}

}  // namespace sfpu
}  // namespace ckernel
