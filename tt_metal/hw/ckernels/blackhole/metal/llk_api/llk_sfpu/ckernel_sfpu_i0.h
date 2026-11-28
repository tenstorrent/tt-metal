// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_polyval.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// i0(x), the modified Bessel function of the first kind, order 0. It is even, so both regions work on |x|:
//   |x| <= 6:       1 + t * P(t), t = x^2, P a minimax fit for relative error
//   |x| >  6:       exp(|x|) / sqrt(|x|) * Q(1/|x|) (A&S 9.7.1), 1/sqrt(2*pi) folded into Q
//   |x| >  91.9008: +inf, the largest float32 x whose I0 is finite; +/-inf gives +inf, NaN gives NaN
// Region 1 runs on every lane and the asymptotic result overwrites it where |x| > 6. Each piece is sized to
// the DEST width (is_fp32_dest_acc_en), not the input format:
//
//                32-bit DEST                                  16-bit DEST
//   P(t)         degree 8                                     degree 5
//   exp(|x|)     _sfpu_exp_fp32_accurate_nonneg_unsafe_<8>    bit pattern of |x|/ln2 + 375
//   1/sqrt(|x|)  magic seed, cubic, one Newton step           magic seed, cubic
//   Q(u)         degree 5                                     degree 1
//
// Both exp forms return exp(|x|) * 2^-8 and Q carries the 2^8, so the result overflows where I0 does rather
// than where exp does (88.72). APPROXIMATION_MODE selects nothing.

// 16-bit DEST asymptotic region: exp(|x|) * 2^-8 * rsqrt_y * Q(1/|x|), rsqrt_y = 1/sqrt(|x|).
inline sfpi::vFloat calculate_i0_asymptotic_bf16_(const sfpi::vFloat abs_x, const sfpi::vFloat rsqrt_y) {
    // w = |x|/ln2 + 375 lies in [375, 512), so its fraction bits shifted left by 8 read as
    // z = 2^(n-8) * m, n = floor(|x|/ln2), m in [1, 2); C(m) = 2^(m-1)/m is a cubic fit (8.7e-4).
    const sfpi::vFloat w = abs_x * 1.442695f + 375.0f;
    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;
    const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(sfpi::exman(w)) << 8);
    const sfpi::vFloat c = PolynomialEvaluator::eval(
        sfpi::setexp(z, 127), 1.775390625f, -1.376953125f, 0.70751953125f, -0.10650634765625f);
    // Q * 2^8: degree 1, 4.9e-4 relative.
    const sfpi::vFloat correction = inv_abs_x * 14.359375f + 102.0625f;
    return z * c * rsqrt_y * correction;
}

// The rsqrt seed and cubic-correction constants, held across the loop.
inline void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstIntPrgm0 = 0x5f1110a0;
    sfpi::vConstFloatPrgm1 = 2.2825186f;
    sfpi::vConstFloatPrgm2 = 2.2533049f;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_i0() {
    // The largest x whose I0 is finite in float32 (0x42b7cd31); on a 16-bit DEST, the largest such fp16 value.
    constexpr float I0_MAX_INPUT = is_fp32_dest_acc_en ? 91.90076446533203f : 91.875f;
    constexpr float I0_THRESHOLD = 6.0f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat abs_x = sfpi::abs(x);

        // Region 1 and 1/sqrt(|x|) on every lane, interleaved to fill SFPMAD latency slots and scoped so their
        // temporaries free LRegs before the asymptotic block. rsqrt: magic seed, then y * (k1 + c * (k2 + c)),
        // c = -|x| * y^2 (2.0e-5 relative); a 32-bit DEST adds one Newton step (7.3e-8).
        sfpi::vFloat val;
        sfpi::vFloat rsqrt_y;
        sfpi::vFloat q;  // Q(1/|x|), 32-bit DEST only
        {
            const sfpi::vFloat t = abs_x * abs_x;
            const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
            if constexpr (is_fp32_dest_acc_en) {
                // Degree 8, 1.0e-8 relative; the t and t^2 terms are the exact Maclaurin 1/4 and 1/64.
                sfpi::vFloat p = t * 1.4044321e-14f + 2.145839e-12f;
                rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - rsqrt_i);
                p = p * t + 4.7780996e-10f;
                sfpi::vFloat c = abs_x * rsqrt_y;
                p = p * t + 6.7723136e-08f;
                c = (-rsqrt_y) * c;
                p = p * t + 6.7822934e-06f;
                sfpi::vFloat k = sfpi::vConstFloatPrgm2 + c;
                p = p * t + 4.3402635e-04f;
                k = c * k + sfpi::vConstFloatPrgm1;
                p = p * t + 0.015625f;
                rsqrt_y = rsqrt_y * k;
                p = p * t + 0.25f;
                c = abs_x * rsqrt_y;
                const sfpi::vFloat half_y = sfpi::addexp(rsqrt_y, -1 /* exp */);
                c = (-rsqrt_y) * c + 1.0f;
                val = p * t + 1.0f;
                rsqrt_y = c * half_y + rsqrt_y;

                // Q(u) * 2^8, u = 1/|x| = rsqrt_y^2: degree 5, 3.4e-8 relative.
                const sfpi::vFloat q5 = 159.0f;
                const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;
                q = PolynomialEvaluator::eval(inv_abs_x, 102.129196f, 12.770073f, 6.99609375f, 11.265625f, -24.25f, q5);
            } else {
                // Degree 5, 1.3e-4 relative.
                sfpi::vFloat p = t * 1.2572855e-07f + 4.7385693e-06f;
                rsqrt_y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - rsqrt_i);
                p = p * t + 4.632473e-04f;
                sfpi::vFloat c = abs_x * rsqrt_y;
                p = p * t + 0.01546478271484375f;
                c = (-rsqrt_y) * c;
                p = p * t + 0.250244140625f;
                sfpi::vFloat k = sfpi::vConstFloatPrgm2 + c;
                val = p * t + 1.0f;
                k = c * k + sfpi::vConstFloatPrgm1;
                rsqrt_y = rsqrt_y * k;
            }
        }

        // Asymptotic region; past I0_MAX_INPUT, abs_x * inf gives +inf and keeps NaN.
        v_if(abs_x > I0_THRESHOLD) {
            if constexpr (is_fp32_dest_acc_en) {
                const sfpi::vFloat scale = rsqrt_y * q;
                val = _sfpu_exp_fp32_accurate_nonneg_unsafe_<8>(abs_x) * scale;
            } else {
                val = calculate_i0_asymptotic_bf16_(abs_x, rsqrt_y);
            }
            v_if(abs_x > I0_MAX_INPUT) { val = abs_x * std::numeric_limits<float>::infinity(); }
            v_endif;
        }
        v_endif;

        // Keyed on the DEST width so a bf16-in/float32-out call keeps float32.
        if constexpr (!is_fp32_dest_acc_en) {
            val = sfpi::convert<sfpi::vFloat16b>(val, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
