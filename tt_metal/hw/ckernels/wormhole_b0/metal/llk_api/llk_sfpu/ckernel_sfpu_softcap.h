// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Newton-Raphson reciprocal: returns 1/|in| (always positive).
// Adapted from _reciprocal_compat_ in ckernel_sfpu_rsqrt_compat.h.
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

// Accurate exp(t) for t >= 0 using Cody-Waite range reduction + degree-7 Horner polynomial.
// Returns exp(t) with ~23-bit mantissa precision.
sfpi_inline sfpi::vFloat softcap_exp(sfpi::vFloat t) {
    constexpr float log2ef = 1.4426950408889634f;
    constexpr float ln2_hi = 6.93145751953125e-1f;
    constexpr float ln2_lo = 1.4286068203094172e-6f;

    // Range reduction: n = round(t / ln2), r = t - n*ln2
    sfpi::vFloat z = t * sfpi::vFloat(log2ef);

    // Round to nearest integer via magic-number add/subtract
    sfpi::vFloat magic = sfpi::vFloat(8388608.0f);  // 2^23
    sfpi::vFloat n_float = (z + magic) - magic;

    // Cody-Waite: compute r = t - n*ln2 accurately (two-part ln2)
    sfpi::vFloat r = t - n_float * sfpi::vFloat(ln2_hi);
    r = r - n_float * sfpi::vFloat(ln2_lo);

    // Horner polynomial for exp(r), degree 7: sum_{k=0}^{7} r^k / k!
    sfpi::vFloat p = sfpi::vFloat(0.00019841270f);   // 1/5040
    p = p * r + sfpi::vFloat(0.0013888889f);          // 1/720
    p = p * r + sfpi::vFloat(0.0083333335f);          // 1/120
    p = p * r + sfpi::vFloat(0.041666668f);           // 1/24
    p = p * r + sfpi::vFloat(0.16666667f);            // 1/6
    p = p * r + sfpi::vFloat(0.5f);                   // 1/2
    p = p * r + sfpi::vConst1;                        // 1
    p = p * r + sfpi::vConst1;                        // 1 + r*(...)

    // Scale by 2^n: exp(t) = exp(r) * 2^n
    sfpi::vInt n_int = _float_to_int32_positive_(n_float);
    sfpi::vFloat two_to_n = sfpi::setexp(sfpi::vConst1, sfpi::vInt(127) + n_int);

    return p * two_to_n;
}

// softcap(x, cap) = cap * tanh(x / cap)
//
// Algorithm:
//   y = x / cap
//   |y| >= 9:     tanh(y) = sign(y)            (saturation)
//   0.1 <= |y| < 9: tanh(y) = 1 - 2/(exp(2|y|) + 1)  (exp-based)
//   |y| < 0.1:    tanh(y) = y - y^3/3 + 2y^5/15 - 17y^7/315  (Taylor)
//   result = cap * tanh(y)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(std::uint32_t param0) {
    // Load cap from bfloat16-encoded parameter
    sfpi::vFloat v_cap = sfpi::s2vFloat16b(param0);
    // Pre-compute 1/cap (24-bit precision via Newton-Raphson)
    sfpi::vFloat v_inv_cap = softcap_recip<3>(v_cap);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // y = x / cap
        sfpi::vFloat y = x * v_inv_cap;
        sfpi::vFloat abs_y = sfpi::setsgn(y, 0);

        // Default: saturation region (|y| >= 9)
        // tanh(y) = sign(y), so result = sign(y) * cap
        sfpi::vFloat tanh_y = sfpi::vConst1;

        // Exp-based region: 0.1 <= |y| < 9
        v_if(abs_y < 9.0f) {
            // t = 2 * |y|
            sfpi::vFloat t = abs_y + abs_y;

            // exp(2|y|) via accurate range-reduced exp
            sfpi::vFloat exp_t = softcap_exp(t);

            // tanh(|y|) = 1 - 2 / (exp(2|y|) + 1)
            sfpi::vFloat denom = exp_t + sfpi::vConst1;
            sfpi::vFloat inv_denom = softcap_recip<3>(denom);
            tanh_y = sfpi::vConst1 - sfpi::vFloat(2.0f) * inv_denom;
        }
        v_endif;

        // Taylor region: |y| < 0.1
        // tanh(y) = y - y^3/3 + 2y^5/15 - 17y^7/315
        v_if(abs_y < 0.1f) {
            sfpi::vFloat y2 = y * y;
            sfpi::vFloat p = sfpi::vFloat(-0.053968254f);  // -17/315
            p = p * y2 + sfpi::vFloat(0.13333334f);        //  2/15
            p = p * y2 + sfpi::vFloat(-0.33333334f);       // -1/3
            p = p * y2 + sfpi::vConst1;                    //  1
            tanh_y = y * p;
        }
        v_endif;

        // For the non-Taylor branches, apply sign of y
        // (Taylor branch already uses signed y)
        v_if(abs_y >= 0.1f) {
            v_if(y < 0.0f) { tanh_y = -tanh_y; }
            v_endif;
        }
        v_endif;

        // result = cap * tanh(y)
        sfpi::vFloat result = v_cap * tanh_y;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {
    // No programmable constants needed
}

}  // namespace ckernel::sfpu
