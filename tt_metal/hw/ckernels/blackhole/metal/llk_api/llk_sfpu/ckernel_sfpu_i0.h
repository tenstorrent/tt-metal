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

// ======================================================================
// i0(x) — modified Bessel function of the first kind, order 0. i0 is even,
// so both regions work on |x|.
//   |x| <= 6:    I0 = 1 + t * P(t), t = x^2, P a minimax fit for relative error.
//   |x| >  6:    I0 = exp(|x|) / sqrt(|x|) * Q(1/|x|)  (Abramowitz & Stegun
//                9.7.1), with 1/sqrt(2*pi) folded into Q's leading term.
//   |x| > 91.9008: +inf (the largest float32 x whose I0 is finite); +/-inf
//                gives +inf and NaN gives NaN.
//
// Region 1 runs for every vector and the asymptotic result overwrites it on
// lanes with |x| > 6. Each piece is sized to the DEST width
// (is_fp32_dest_acc_en, not the input format: a bf16-in/float32-out call
// runs a 32-bit DEST):
//
//                32-bit DEST                      16-bit DEST
//   P(t)         degree 8 in t                    degree 5 in t
//   exp(|x|)     _sfpu_exp_fp32_accurate_         bit pattern of |x|/ln2 + 375,
//                nonneg_unsafe_<8>                cubic correction
//   1/sqrt(|x|)  magic seed, cubic correction,    magic seed, cubic correction
//                one Newton step
//   Q(u)         degree 5                         degree 1
//
// exp(|x|) overflows float32 at 88.72, before I0 does, so both columns form
// exp(|x|) * 2^-8 and carry the 2^8 in Q's coefficients: a power-of-two scale
// changes no rounding, and the result overflows only where I0 does.
//
// Coefficients are bf16- or fp16-exact where accuracy allows, so each loads
// with one SFPLOADI instead of two; the rsqrt constants live in the
// programmable constant registers (set in i0_init); and the rsqrt chain is
// interleaved with region 1's Horner chain to fill SFPMAD latency slots.
//
// APPROXIMATION_MODE is accepted for call-site compatibility and selects
// nothing.
// ======================================================================

// 16-bit DEST: exp(|x|) * rsqrt_y * Q(rsqrt_y^2), with rsqrt_y = 1/sqrt(|x|)
// from the caller, ordered so independent work fills SFPMAD latency slots.
// 1/|x| = (1/sqrt(|x|))^2 reuses the rsqrt instead of a fresh reciprocal;
// Q(u) is fit for relative error on u in [1/91.9, 1/6]. The 32-bit DEST
// counterpart is written out in calculate_i0.
inline sfpi::vFloat calculate_i0_asymptotic_bf16_(const sfpi::vFloat abs_x, const sfpi::vFloat rsqrt_y) {
    // exp(|x|) * 2^-8 from the bit pattern of w = |x|/ln2 + 375: for |x| in
    // [0, 94.9], w lies in [375, 512), so its exponent is fixed and its 23
    // fraction bits hold (|x|/ln2 + 119) * 2^15. Shifted left by 8 they
    // read as a float32 z = 2^(n-8) * (1 + r), n = floor(|x|/ln2), r the
    // fraction to 15 bits -- the integer split costs two instructions.
    // exp(|x|) * 2^-8 = z * C(m), C(m) = 2^(m-1)/m with m = 1 + r, a cubic
    // fit (8.7e-4 relative, all four coefficients fp16-exact).
    const sfpi::vFloat w = abs_x * 1.442695f + 375.0f;
    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;
    const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(sfpi::exman(w)) << 8);
    const sfpi::vFloat c = PolynomialEvaluator::eval(
        sfpi::setexp(z, 127), 1.775390625f, -1.376953125f, 0.70751953125f, -0.10650634765625f);
    // Q * 2^8: degree 1, fit 4.9e-4 relative, both coefficients fp16-exact.
    const sfpi::vFloat correction = inv_abs_x * 14.359375f + 102.0625f;
    return z * c * rsqrt_y * correction;
}

// The rsqrt seed constant and its cubic-correction constants, shared by both
// DEST widths. Programmable registers hold them across the loop, where each
// would otherwise be an SFPLOADI pair per iteration.
inline void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstIntPrgm0 = 0x5f1110a0;
    sfpi::vConstFloatPrgm1 = 2.2825186f;
    sfpi::vConstFloatPrgm2 = 2.2533049f;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_i0() {
    // The largest x whose I0 is finite in float32 (0x42b7cd31). A 16-bit DEST
    // holds bf16 or fp16 inputs, whose largest such value, 91.875, loads in
    // one SFPLOADI.
    constexpr float I0_MAX_INPUT = is_fp32_dest_acc_en ? 91.90076446533203f : 91.875f;
    constexpr float I0_THRESHOLD = 6.0f;

#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat abs_x = sfpi::abs(x);

        // ─── Region 1 (always; valid for |x| <= 6) and 1/sqrt(|x|) ──────────
        // Computed unconditionally, in a nested scope so their temporaries
        // are freed before the asymptotic block needs the LRegs.
        //
        // P is evaluated by Horner (p) with the 1/sqrt(|x|) chain (rsqrt_y)
        // interleaved statement by statement.
        // 1/sqrt(|x|): magic seed vConstIntPrgm0 - (bits >> 1), then the cubic
        // correction y * (k1 + c * (k2 + c)), c = -|x| * y^2 (2.0e-5 relative);
        // a 32-bit DEST adds one Newton step (7.3e-8).
        sfpi::vFloat val;
        sfpi::vFloat rsqrt_y;
        sfpi::vFloat q;  // Q(1/|x|), 32-bit DEST only
        {
            const sfpi::vFloat t = abs_x * abs_x;
            const sfpi::vInt rsqrt_i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(abs_x) >> 1);
            if constexpr (is_fp32_dest_acc_en) {
                // Degree 8 in t (9 terms incl. the leading 1), 1.0e-8 relative.
                // k = 1, 2 are the exact Maclaurin values 1/4 and 1/64; k = 7, 8
                // are bf16-exact, with the rest refit around them.
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

                // Q(u) * 2^8, u = 1/|x| = rsqrt_y^2 in [1/91.9, 1/6]: degree 5, 3.4e-8
                // relative. Evaluated ahead of the |x| > 6 compare, which then fills
                // the latency slot before the multiply by rsqrt_y.
                const sfpi::vFloat q5 = 159.0f;
                const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;
                q = PolynomialEvaluator::eval(inv_abs_x, 102.129196f, 12.770073f, 6.99609375f, 11.265625f, -24.25f, q5);
            } else {
                // Degree 5 in t, 1.3e-4 relative, every coefficient bf16- or
                // fp16-exact: bf16 output cannot see the terms the 32-bit fit
                // carries beyond it.
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

        // ─── Asymptotic overwrite for |x| > 6; +inf past 91.9 ─────────────
        // Both exp constructions assume |x| < 93.9; lanes above 91.9 are
        // overwritten here. abs_x * inf maps +/-inf to +inf and keeps NaN.
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
