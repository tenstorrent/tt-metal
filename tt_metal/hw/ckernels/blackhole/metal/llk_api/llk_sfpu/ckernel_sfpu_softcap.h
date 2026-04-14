// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
//
// Implementation:
//   u = x / cap  (precomputed as x * inv_cap)
//   Segment A (|u| <= 0.625): degree-13 Taylor polynomial for tanh
//   Segment B (0.625 < |u| <= 9): exp-based: (exp(2|u|)-1) * recip(exp(2|u|)+1)
//   Segment C (|u| > 9): saturate to sign(u)
//   result = cap * tanh(u)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t param0) {
    union {
        uint32_t u;
        float f;
    } cap_bits;
    cap_bits.u = param0;
    const float cap = cap_bits.f;
    const float inv_cap = 1.0f / cap;

    // Exp range reduction constants
    constexpr float log2e = 1.4426950408889634f;
    // Split ln2 into high + low parts for precise range reduction
    constexpr float ln2_hi = 0.693145751953125f;      // 13-bit exact
    constexpr float ln2_lo = 1.4286068203094172e-6f;  // correction
    constexpr float magic_round = 12582912.0f;        // 1.5 * 2^23
    constexpr float magic_int = 8388608.0f;           // 2^23

    // Taylor coefficients for tanh(u)/u = 1 + c3*s + c5*s^2 + ... where s = u^2
    // Extended to degree-21 for boundary at 0.7
    constexpr float c3 = -0.33333333333333333f;      // -1/3
    constexpr float c5 = 0.13333333333333333f;       //  2/15
    constexpr float c7 = -0.053968253968253968f;     // -17/315
    constexpr float c9 = 0.021869488536155203f;      //  62/2835
    constexpr float c11 = -0.0088632355299021856f;   // -1382/155925
    constexpr float c13 = 0.0035921280365724804f;    //  21844/6081075
    constexpr float c15 = -0.0014558343870513183f;   // -929569/638512875
    constexpr float c17 = 0.00059002744094558616f;   //  6404582/10854718875
    constexpr float c19 = -0.00023912911424354545f;  // -443861162/1856156927625
    constexpr float c21 = 0.000096940033741907956f;  //  18888223693065/...

    constexpr float taylor_bound = 0.74f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat u = x * inv_cap;
        sfpi::vFloat au = sfpi::abs(u);

        // Start with saturated value: tanh = sign(u) * 1.0
        // result = sign(x) * cap  (cap > 0)
        sfpi::vFloat result = sfpi::setsgn(sfpi::vFloat(cap), x);

        // Segment B: 0.625 < |u| <= 9: exp-based tanh
        v_if(au <= 9.0f) {
            sfpi::vFloat t = au + au;  // 2 * |u|

            // Range reduction: n = round(t * log2e), r = t - n * ln2
            sfpi::vFloat s = t * log2e;
            sfpi::vFloat s_biased = s + magic_round;
            sfpi::vFloat n_float = s_biased - magic_round;
            sfpi::vFloat r = t - n_float * ln2_hi - n_float * ln2_lo;

            // exp(r) via degree-6 Horner polynomial
            // Fewer additions means less accumulated rounding error
            sfpi::vFloat exp_r = r * (1.0f / 720.0f) + (1.0f / 120.0f);
            exp_r = exp_r * r + (1.0f / 24.0f);
            exp_r = exp_r * r + (1.0f / 6.0f);
            exp_r = exp_r * r + 0.5f;
            exp_r = exp_r * r + 1.0f;
            exp_r = exp_r * r + 1.0f;

            // 2^n via exponent manipulation
            sfpi::vFloat temp = n_float + magic_int;
            sfpi::vUInt n_bits = sfpi::reinterpret<sfpi::vUInt>(temp);
            sfpi::vUInt n_uint = n_bits & 0x7FFFFF;
            sfpi::vUInt biased_exp = n_uint + 127;
            sfpi::vFloat two_pow_n = sfpi::setexp(sfpi::vConst1, biased_exp);

            // exp(2|u|) = exp(r) * 2^n
            sfpi::vFloat exp_t = exp_r * two_pow_n;

            // tanh(|u|) = 1 - 2 / (exp_t + 1)
            sfpi::vFloat den = exp_t + 1.0f;

            // Newton-Raphson reciprocal of den
            sfpi::vUInt den_bits = sfpi::reinterpret<sfpi::vUInt>(den);
            sfpi::vFloat recip = sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(0x7F000000u) - den_bits);
            recip = recip * (2.0f - den * recip);
            recip = recip * (2.0f - den * recip);
            recip = recip * (2.0f - den * recip);

            sfpi::vFloat tanh_au = 1.0f - 2.0f * recip;

            // result = sign(u) * cap * tanh(|u|)
            result = sfpi::setsgn(tanh_au * cap, x);
        }
        v_endif;

        // Segment A: |u| <= 0.74: degree-21 Taylor polynomial (overrides segment B)
        v_if(au <= taylor_bound) {
            sfpi::vFloat s = u * u;
            sfpi::vFloat poly = s * c21 + c19;
            poly = poly * s + c17;
            poly = poly * s + c15;
            poly = poly * s + c13;
            poly = poly * s + c11;
            poly = poly * s + c9;
            poly = poly * s + c7;
            poly = poly * s + c5;
            poly = poly * s + c3;
            poly = poly * s + 1.0f;
            // tanh(u) = u * poly; result = cap * u * poly = x * poly
            result = x * poly;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init() {}

}  // namespace sfpu
}  // namespace ckernel
