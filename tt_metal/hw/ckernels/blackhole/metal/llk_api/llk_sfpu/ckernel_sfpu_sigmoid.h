// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

// exp_constants supplies the constants of the FP32 exponential (ExpFp32Constants, or ExpFp32PreloadedConstants the
// caller loads once before its row loop); the BF16 path does not use it.
template <bool is_fp32_acc_to_dest_mode = true, typename ExpConstants = ExpFp32Constants>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x, const ExpConstants& exp_constants = ExpConstants{}) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_fp32_accurate_(-x, exp_constants);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
    }

    sfpi::vFloat denominator = 1.0f + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = sfpu_reciprocal_iter<2>(denominator);
    } else {
        result = sfpu_reciprocal_iter<1>(denominator);
    }

    return result;
}

// The BF16-grade sigmoid of _sfpu_sigmoid_<false> (exp_21f of -x, then one Newton step of the reciprocal of
// 1 + exp(-x)), as computed by _sfpu_sigmoid_bf16_rows_ below.
// (-x) / ln2 is formed as x * -(1 / ln2): the product is the same, sign included. The exponent is clamped at 254
// rather than 255, so exp(-x) stays finite (below 2**128) and the reciprocal needs no NaN guard: from 2**126 up its
// estimate is 0 and the Newton step keeps it 0, the result the guarded step gives for an infinite exp(-x).
//
// exp_21f splits the clamped t = x * -(1/ln2) + 127, in [0, 254], into its integer part n (the result's exponent) and
// its fraction, scaled by 2**23 for the polynomial. Here n = trunc(t) is one round-to-zero SFPSTOCHRND and the
// fraction t - n one exact subtraction, instead of shifting t's mantissa into an integer and splitting that again
// (six instructions down to three). The polynomial takes the unscaled fraction with c1 and c2 scaled by 2**23 and
// 2**46 instead: Horner's steps then compute 2**23 times, and finally exactly, the values of the scaled form, so each
// rounds the same. Both splits agree wherever t >= 1. Below 1 (and at the three inputs for which round-to-zero
// SFPSTOCHRND rounds up, 0x3f7ffffe, 0x3f7fffff and 0x3fffffff), exp(-x) is below 2**-124 in either form, so
// 1 + exp(-x) rounds to 1 in both.
// The Newton step forms y * (2 - d * y) directly: 2 - d * y is -(d * y - 2) exactly, and y * (2 - d * y) is never
// -0 (y >= +0, and 2 - d * y is 2 where y is 0), so adding no -0 leaves it unchanged.
// Constants: -(1/ln2), the exponent bias 127, c0, c1 (EXP_21F_BF16_C1 * 2**23), c2 (EXP_21F_BF16_C2 * 2**46), and
// the clamp's upper bound t_max = 254.
constexpr float EXP_21F_BF16_C1_UNSCALED = EXP_21F_BF16_C1 * 0x1p23f;
constexpr float EXP_21F_BF16_C2_UNSCALED = EXP_21F_BF16_C2 * 0x1p46f;

// -(1/ln2), which _sfpu_sigmoid_bf16_rows_'s caller loads once before its row loop, and the clamp's upper bound.
struct SigmoidBf16Constants {
    sfpi::vFloat neg_one_ln2 = -1.4426950216293334961f;
    static constexpr float t_max = 254.0f;
};

// _sfpu_sigmoid_bf16_preloaded_ of N independent rows, step by step across the rows: the single-row chain stalls on
// nearly every dependent MAD-unit result (two-cycle latency), so the second row fills those cycles. The same
// operations as _sfpu_sigmoid_bf16_preloaded_ on each row, so the result is bit-identical. Two rows leave no lregs to
// keep the constants in, so each but -(1/ln2) (the eighth lreg, which the caller loads once before its row loop) is
// loaded where it is used, one load serving both rows. With constants_in_cregs the clamp bound t_max and c2 come from
// vConstFloatPrgm1 and vConstFloatPrgm2 (silu_init). Requires vConstFloatPrgm0 = 2.0.
template <bool constants_in_cregs, int N>
sfpi_inline void _sfpu_sigmoid_bf16_rows_(sfpi::vFloat (&x)[N], const SigmoidBf16Constants& constants) {
    static_assert(N == 1 || N == 2, "two rows use the eight lregs");
    using K = SigmoidBf16Constants;
    sfpi::vFloat t[N], f[N];
    sfpi::vUInt8 n[N];
    sfpi::vFloat c = 127.f;
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        t[i] = x[i] * constants.neg_one_ln2 + c;
    }
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        if constexpr (constants_in_cregs) {
            t[i] = sfpi::min(t[i], sfpi::vFloat(sfpi::vConstFloatPrgm1));
        } else {
            t[i] = sfpi::min(t[i], sfpi::vFloat(K::t_max));
        }
    }
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        t[i] = sfpi::max(t[i], 0.0f);
    }
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        n[i] = sfpi::convert<sfpi::vUInt8>(t[i], sfpi::RoundMode::Zero);
    }
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        f[i] = t[i] - sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(n[i]), sfpi::RoundMode::Nearest);
    }
    // c0 + f * (c1 + f * c2), as PolynomialEvaluator::eval.
    c = EXP_21F_BF16_C1_UNSCALED;
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        if constexpr (constants_in_cregs) {
            t[i] = f[i] * sfpi::vConstFloatPrgm2 + c;
        } else {
            t[i] = f[i] * EXP_21F_BF16_C2_UNSCALED + c;
        }
    }
    c = EXP_21F_BF16_C0;
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        t[i] = c + f[i] * t[i];
    }
    // d = 1 + exp(-x) (into f), y = its reciprocal estimate refined by one Newton step (into x).
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        f[i] = sfpi::setexp(t[i], sfpi::as<sfpi::vInt>(n[i])) + 1.0f;
    }
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        x[i] = sfpi::approx_recip(f[i]);
    }
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        t[i] = sfpi::vConstFloatPrgm0 - f[i] * x[i];
    }
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        x[i] = x[i] * t[i];
    }
}

// The constant of _sfpu_sigmoid_fp32_rows_ its caller loads once before the row loop (the eighth lreg).
struct SigmoidFp32Constants {
    sfpi::vFloat neg_one_ln2 = -ExpFp32Constants::one_ln2;
    // The bound of |x|, and the bits of 1.5 * 2**23 + 127, which j + 127 lands in.
    static constexpr float clamp = 88.0f;
    static constexpr std::uint32_t exp2_bias_bits = 0x4B400000U + 127U;
};

// sigmoid(x) = 1 / (1 + exp(-x)) of N independent rows, step by step across the rows, so each row's dependent
// instructions fill the latency of the others' (the single-row chain is latency-bound on Blackhole). Two rows fit
// seven lregs with the other constants reloaded where they are used, and each coefficient load serves both rows.
// Bit-identical to _sfpu_sigmoid_<true> (exp_fp32_accurate of -x, then two Newton steps of the reciprocal of
// 1 + exp(-x)) for every finite and infinite x, in fewer instructions per row:
// - |x| is clamped to 88 first. For x < -88, exp(-x) > 1.6e38 >= 2**126, where the reciprocal estimate, and so the
//   result, is +0 exactly as for any larger exp(-x) or its overflow to infinity. For x > 87.4, exp(-x) < 2**-126, so
//   1 + exp(-x) is 1 exactly as for any smaller exp(-x), which the exponential flushes to 0. Within the clamp exp(-x)
//   needs no overflow or underflow guard, and 1 + exp(-x) is finite, so the Newton steps need no NaN guard either.
// - -x is folded into the range reduction: j = x * -(1/ln2) is the product (-x) * (1/ln2), and j * -ln2_hi - x is
//   j * -ln2_hi + (-x), each rounded once as before.
// - 2**j (j in [-127, 127] within the clamp) is built from j's bits instead of converting j to two's complement and
//   adding it to exp(f)'s exponent: exp(f) * 2**j is exact while it is normal, and 1 + exp(f) * 2**j is one MAD, the
//   same value as 1 + setexp(exp(f), e). Where that exponent e is below 1 (always at j = -127, whose bits give
//   2**j = +0) the exponential flushed to 0, and 1 + exp(f) * 2**j is 1 too.
// - 2 - d * y is formed directly (rounded to nearest, it is -(d * y - 2) exactly), and y * (2 - d * y) is never -0,
//   so the Newton steps add +0 rather than load -0.
// A +NaN x clamps to 88, giving 1; a -NaN x gives NaN (the unclamped form gives 0 for either NaN).
// Requires vConstFloatPrgm0 = 2.0 (sigmoid_init). With constants_in_cregs, the clamp bound 88 and the 2**j bias come
// from vConstFloatPrgm1 and vConstFloatPrgm2 (sigmoid_init) rather than SFPLOADIs: four instructions fewer per pair
// of rows, as the swap against a constant register leaves the bound in place, where it overwrites a loaded one.
template <bool constants_in_cregs, int N>
sfpi_inline void _sfpu_sigmoid_fp32_rows_(sfpi::vFloat (&x)[N], const SigmoidFp32Constants& constants) {
    static_assert(N == 1 || N == 2, "two rows use seven of the eight lregs");
    using C = ExpFp32Constants;
    sfpi::vFloat j[N], f[N], s[N], r[N];
    // x = copysgn(min(|x|, 88), x), one swap per row.
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        j[n] = sfpi::abs(x[n]);
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        if constexpr (constants_in_cregs) {
            j[n] = sfpi::min(j[n], sfpi::vFloat(sfpi::vConstFloatPrgm1));
        } else {
            j[n] = sfpi::min(j[n], sfpi::vFloat(SigmoidFp32Constants::clamp));
        }
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        x[n] = sfpi::copysgn(j[n], x[n]);
    }

    // j = round(-x / ln2).
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        j[n] = x[n] * constants.neg_one_ln2;
    }
    sfpi::vSMag16 sm[N];
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        sm[n] = sfpi::convert<sfpi::vSMag16>(j[n], sfpi::RoundMode::Nearest);
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        j[n] = sfpi::convert<sfpi::vFloat>(sm[n], sfpi::RoundMode::Nearest);
    }

    // f = -x - j * ln2 (two-part Cody-Waite).
    sfpi::vFloat k = C::neg_ln2_hi;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        f[n] = j[n] * k - x[n];
    }
    k = C::neg_ln2_lo;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        f[n] = j[n] * k + f[n];
    }

    // 2**j: j + 127 lands in the low mantissa bits of j + (1.5 * 2**23 + 127); shifted into the exponent field.
    if constexpr (constants_in_cregs) {
#pragma GCC unroll 2
        for (int n = 0; n < N; n++) {
            s[n] = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(j[n] + sfpi::vConstFloatPrgm2) << 23);
        }
    } else {
        k = Converter::as_float(SigmoidFp32Constants::exp2_bias_bits);
#pragma GCC unroll 2
        for (int n = 0; n < N; n++) {
            s[n] = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(j[n] + k) << 23);
        }
    }

    // exp(f) on [-ln2/2, ln2/2], the degree-6 minimax of _sfpu_exp_fp32_accurate_ in the same Horner order.
    k = C::c5;
    sfpi::vFloat c = C::c4;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = k * f[n] + c;
    }
    c = C::c3;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + c;
    }
    c = C::c2;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + c;
    }
    c = C::c1;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + c;
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + 1.0f;
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + 1.0f;
    }

    // d = 1 + exp(-x) in [1, 2**128) (reusing f), then two Newton steps of 1 / d (into s and x).
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        f[n] = r[n] * s[n] + 1.0f;
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        s[n] = sfpi::approx_recip(f[n]);
    }
#pragma GCC unroll 2
    for (int i = 0; i < 2; i++) {
#pragma GCC unroll 2
        for (int n = 0; n < N; n++) {
            r[n] = sfpi::vConstFloatPrgm0 - f[n] * s[n];
        }
#pragma GCC unroll 2
        for (int n = 0; n < N; n++) {
            s[n] = s[n] * r[n];
        }
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        x[n] = s[n];
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        // The BF16-grade sigmoid, two rows at a time. Each result is rounded to BF16 as a vFloat, so the store keeps
        // the destination's format.
        const SigmoidBf16Constants constants;
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS / 2; d++) {
            sfpi::vFloat rows[2] = {sfpi::dst_reg[0], sfpi::dst_reg[1]};
            _sfpu_sigmoid_bf16_rows_</*constants_in_cregs=*/false>(rows, constants);
            sfpi::dst_reg[0] = sfpi::vFloat(sfpi::convert<sfpi::vFloat16b>(rows[0], sfpi::RoundMode::Nearest));
            sfpi::dst_reg[1] = sfpi::vFloat(sfpi::convert<sfpi::vFloat16b>(rows[1], sfpi::RoundMode::Nearest));
            sfpi::dst_reg += 2;
        }
        if constexpr (ITERATIONS % 2 != 0) {
            sfpi::vFloat row[1] = {sfpi::dst_reg[0]};
            _sfpu_sigmoid_bf16_rows_</*constants_in_cregs=*/false>(row, constants);
            sfpi::dst_reg[0] = sfpi::vFloat(sfpi::convert<sfpi::vFloat16b>(row[0], sfpi::RoundMode::Nearest));
            sfpi::dst_reg++;
        }
    } else if constexpr (!APPROXIMATION_MODE) {
        // The FP32-accurate sigmoid, two rows at a time.
        const SigmoidFp32Constants constants;
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS / 2; d++) {
            sfpi::vFloat rows[2] = {sfpi::dst_reg[0], sfpi::dst_reg[1]};
            _sfpu_sigmoid_fp32_rows_</*constants_in_cregs=*/true>(rows, constants);
            sfpi::dst_reg[0] = rows[0];
            sfpi::dst_reg[1] = rows[1];
            sfpi::dst_reg += 2;
        }
        if constexpr (ITERATIONS % 2 != 0) {
            sfpi::vFloat row[1] = {sfpi::dst_reg[0]};
            _sfpu_sigmoid_fp32_rows_</*constants_in_cregs=*/true>(row, constants);
            sfpi::dst_reg[0] = row[0];
            sfpi::dst_reg++;
        }
    } else {
        calculate_sigmoid_appx<ITERATIONS>();
    }
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE) {
        sfpu_reciprocal_init<false>();
        // The FP32 path's constants (_sfpu_sigmoid_fp32_rows_<true>); the BF16 path uses neither.
        sfpi::vConstFloatPrgm1 = SigmoidFp32Constants::clamp;
        sfpi::vConstFloatPrgm2 = Converter::as_float(SigmoidFp32Constants::exp2_bias_bits);
    } else {
        sigmoid_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
