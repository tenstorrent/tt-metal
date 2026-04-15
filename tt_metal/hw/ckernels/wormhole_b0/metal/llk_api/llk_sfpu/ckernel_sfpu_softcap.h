// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Newton-Raphson reciprocal: returns 1/|in| (always positive).
template <int max_iter = 3>
sfpi_inline sfpi::vFloat softcap_recip(const sfpi::vFloat in) {
    sfpi::vFloat val = sfpi::setsgn(in, 1);
    val = setexp(val, 126);
    sfpi::vFloat c = 1.442695f;
    sfpi::vFloat two = 2.0f;
    sfpi::vFloat result = c * (val * c + two);
    for (int s_iter = 0; s_iter < (max_iter - 1); s_iter++) {
        result = result * (val * result + two);
    }
    sfpi::vInt orig_exp = exexp(in);
    sfpi::vInt new_exp = exexp(result);
    new_exp -= orig_exp;
    new_exp += 126;
    v_if(new_exp < 0) {
        result = 0.0F;
        new_exp = 0;
    }
    v_endif;
    return setexp(result, new_exp);
}

// Accurate exp(t) for t >= 0 using Cody-Waite range reduction.
// Uses Hacker's Delight branchless integer rounding + exponent manipulation.
sfpi_inline sfpi::vFloat softcap_exp(sfpi::vFloat t) {
    constexpr float log2ef = 1.4426950408889634f;
    constexpr float neg_ln2_hi = -6.93145751953125e-1f;
    constexpr float neg_ln2_lo = -1.4286068203094172e-6f;

    sfpi::vFloat z = t * sfpi::vFloat(log2ef);

    // Hacker's Delight round-to-nearest-even (works for z >= 0)
    const sfpi::vFloat c231 = sfpi::vFloat(12582912.0f);
    sfpi::vFloat tmp = z + c231;
    sfpi::vFloat k = tmp - c231;
    sfpi::vInt k_int = sfpi::reinterpret<sfpi::vInt>(tmp) - sfpi::reinterpret<sfpi::vInt>(c231);

    // Cody-Waite: r = t - k*ln2 (negated constants for MAD optimization)
    sfpi::vFloat r = k * sfpi::vFloat(neg_ln2_hi) + t;
    r = k * sfpi::vFloat(neg_ln2_lo) + r;

    // Horner polynomial for exp(r), degree 7
    sfpi::vFloat p = sfpi::vFloat(0.00019841270f);
    p = p * r + sfpi::vFloat(0.0013888889f);
    p = p * r + sfpi::vFloat(0.0083333335f);
    p = p * r + sfpi::vFloat(0.041666668f);
    p = p * r + sfpi::vFloat(0.16666667f);
    p = p * r + sfpi::vFloat(0.5f);
    p = p * r + sfpi::vConst1;
    p = p * r + sfpi::vConst1;

    // Scale by 2^k via exponent adjustment
    sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
    return sfpi::setexp(p, p_exp + k_int);
}

// softcap(x, cap) = cap * tanh(x / cap)
// param0 = cap (full fp32 bit pattern), param1 = 1/cap (full fp32 bit pattern)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(std::uint32_t param0, std::uint32_t param1) {
    // Load full-precision fp32 via bit reinterpretation of vInt broadcast
    sfpi::vFloat v_cap = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(param0));
    sfpi::vFloat v_inv_cap = sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(param1));

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat y = x * v_inv_cap;
        sfpi::vFloat abs_y = sfpi::setsgn(y, 0);

        // Default: saturation (|y| >= 9) -> tanh = sign(y)
        sfpi::vFloat tanh_y = sfpi::vConst1;

        // Exp-based region: 0.6 <= |y| < 9
        // tanh(|y|) = 1 - 2/(exp(2|y|) + 1)
        v_if(abs_y < 9.0f) {
            sfpi::vFloat t = abs_y + abs_y;
            sfpi::vFloat exp_t = softcap_exp(t);
            sfpi::vFloat denom = exp_t + sfpi::vConst1;
            sfpi::vFloat inv_denom = softcap_recip<3>(denom);
            tanh_y = sfpi::vConst1 - sfpi::vFloat(2.0f) * inv_denom;
        }
        v_endif;

        // Sollya minimax polynomial region: |y| < 0.6
        // tanh(y) = y * P(y^2) with optimized coefficients
        v_if(abs_y < 0.6f) {
            sfpi::vFloat y2 = y * y;
            sfpi::vFloat p = sfpi::vFloat(1.5497928e-2f);
            p = p * y2 + sfpi::vFloat(-5.2119765e-2f);
            p = p * y2 + sfpi::vFloat(0.13310669f);
            p = p * y2 + sfpi::vFloat(-0.33332360f);
            p = p * y2 + sfpi::vFloat(0.99999994f);
            tanh_y = y * p;
        }
        v_endif;

        // Apply sign for non-polynomial regions
        v_if(abs_y >= 0.6f) {
            v_if(y < 0.0f) { tanh_y = -tanh_y; }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[0] = v_cap * tanh_y;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {}

}  // namespace ckernel::sfpu
