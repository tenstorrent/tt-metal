// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_log.h"
#include "sfpi.h"

namespace ckernel::sfpu {

//
// log1p(x) = log(1 + x), numerically stable for |x| << 1.
//
// Strategy:
//   If |x| < 0.5  → use the identity
//       log1p(x) = x - x²/2 + x³/3 - ...
//   But rather than a raw Taylor series (poor convergence) we use the
//   hardware log after argument reduction:
//       log1p(x) = log(1 + x)   when  |x| >= threshold
//       log1p(x) via compensated log when |x| is small
//
//   The key insight: the hardware _calculate_log_body_no_init_ computes
//   log(m * 2^e) using the mantissa.  For 1+x where x is tiny, the
//   mantissa of (1+x) in fp32 carries x correctly as long as |x| > eps.
//   So for |x| >= 2^{-23} we can just call log(1+x) directly.
//   For x == 0 the result is exactly 0.
//
//   The pathological region is 2^{-23} <= |x| <= 0.5 where log(1+x)
//   would lose digits to the leading 1.  We handle this with the
//   Kahan / Higham compensated formula:
//
//       u = 1.0f + x
//       log1p(x) = log(u) + (x - (u - 1.0f)) / u
//
//   The correction term (x - (u - 1)) captures the rounding that
//   occurred when forming u = 1 + x, recovering the lost low bits.
//   This gives ~1 ulp accuracy across the whole range.
//
// Special cases (handled by the outer op infrastructure):
//   x == 0  → 0
//   x == -1 → -inf
//   x < -1  → NaN
//   x == +inf → +inf
//

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_log1p(const uint dst_offset) {
    for (int d = 0; d < ITERATIONS; ++d) {
        // Load input
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 0);
        sfpi::vFloat x = sfpi::dst_reg[0];

        // u = 1 + x  (may lose low-order bits of x)
        sfpi::vFloat u = sfpi::vConst1 + x;

        // log(u) using existing log body
        sfpi::vFloat log_u = _calculate_log_body_no_init_(u);

        // correction = (x - (u - 1)) / u
        // u - 1 may differ from x due to fp rounding
        sfpi::vFloat u_m1 = u - sfpi::vConst1;
        sfpi::vFloat correction = (x - u_m1);

        // Divide correction by u:  multiply by 1/u
        sfpi::vFloat inv_u = sfpi::vConst1 / u;  // reciprocal
        correction = correction * inv_u;

        sfpi::vFloat result = log_u + correction;

        // Handle x == 0 exactly → 0
        v_if(x == sfpi::vConst0) {
            result = sfpi::vConst0;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 0);
        dst_offset_++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline sfpi::vFloat _calculate_log1p_body_(sfpi::vFloat x) {
    // u = 1 + x
    sfpi::vFloat u = sfpi::vConst1 + x;

    // Compute log(u) via the existing log helper
    sfpi::vFloat log_u = _calculate_log_body_no_init_(u);

    // Kahan / Higham correction: recover bits lost when forming u = 1 + x
    sfpi::vFloat u_m1 = u - sfpi::vConst1;
    sfpi::vFloat diff = x - u_m1;

    // We need 1/u.  Use the SFPU reciprocal.
    sfpi::vFloat inv_u;
    if constexpr (APPROXIMATION_MODE) {
        inv_u = sfpi::approx_recip(u);
    } else {
        inv_u = sfpi::vConst1;
        // Two Newton-Raphson steps starting from hardware reciprocal estimate
        sfpi::vFloat r = sfpi::approx_recip(u);
        // r1 = r * (2 - u*r)
        r = r * (sfpi::vFloat(2.0f) - u * r);
        // r2 = r * (2 - u*r)
        r = r * (sfpi::vFloat(2.0f) - u * r);
        inv_u = r;
    }

    sfpi::vFloat correction = diff * inv_u;
    sfpi::vFloat result = log_u + correction;

    // x == 0 → 0
    v_if(x == sfpi::vConst0) {
        result = sfpi::vConst0;
    }
    v_endif;

    return result;
}

}  // namespace ckernel::sfpu
