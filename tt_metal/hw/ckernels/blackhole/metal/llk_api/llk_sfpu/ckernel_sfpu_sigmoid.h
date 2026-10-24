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
// 1 + exp(-x)), with its constants in registers the caller loads once per call, so its loop reloads no FP32 constant.
// (-x) / ln2 is formed as x * -(1 / ln2): the product is the same, sign included. The exponent is clamped at 254
// rather than 255, so exp(-x) stays finite (below 2**128) and the reciprocal needs no NaN guard: from 2**126 up its
// estimate is 0 and the Newton step keeps it 0, the result the guarded step gives for an infinite exp(-x).
struct SigmoidBf16Constants {
    sfpi::vFloat neg_one_ln2 = -1.4426950216293334961f;
    sfpi::vFloat exp_bias = 127.f;
    sfpi::vFloat c0 = EXP_21F_BF16_C0;
    sfpi::vFloat c1 = EXP_21F_BF16_C1;
    sfpi::vFloat c2 = EXP_21F_BF16_C2;
};

template <typename Constants>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_bf16_preloaded_(sfpi::vFloat x, const Constants& k) {
    sfpi::vFloat xlog2 = sfpi::clamp(x * k.neg_one_ln2 + k.exp_bias, 0.0f, 254.0f);
    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
    sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part = sfpi::exman(z);
    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
    frac = PolynomialEvaluator::eval(frac, k.c0, k.c1, k.c2);
    sfpi::vFloat exp_neg_x = sfpi::setexp(frac, exponential_part);
    return sfpu_reciprocal_iter<1, /*normalized=*/true>(1.0f + exp_neg_x);
}

// The constant of _sfpu_sigmoid_fp32_rows_ its caller loads once before the row loop (the eighth lreg).
struct SigmoidFp32Constants {
    sfpi::vFloat neg_one_ln2 = -ExpFp32Constants::one_ln2;
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
// Requires vConstFloatPrgm0 = 2.0 (sigmoid_init).
template <int N>
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
        j[n] = sfpi::min(j[n], sfpi::vFloat(88.0f));
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
    k = Converter::as_float(0x4B400000U + 127U);
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        s[n] = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(j[n] + k) << 23);
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
        const SigmoidBf16Constants constants;
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat result = _sfpu_sigmoid_bf16_preloaded_(sfpi::dst_reg[0], constants);
            // Rounded to BF16 as a vFloat, so the store keeps the destination's format.
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else if constexpr (!APPROXIMATION_MODE) {
        // The FP32-accurate sigmoid, two rows at a time.
        const SigmoidFp32Constants constants;
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS / 2; d++) {
            sfpi::vFloat rows[2] = {sfpi::dst_reg[0], sfpi::dst_reg[1]};
            _sfpu_sigmoid_fp32_rows_(rows, constants);
            sfpi::dst_reg[0] = rows[0];
            sfpi::dst_reg[1] = rows[1];
            sfpi::dst_reg += 2;
        }
        if constexpr (ITERATIONS % 2 != 0) {
            sfpi::vFloat row[1] = {sfpi::dst_reg[0]};
            _sfpu_sigmoid_fp32_rows_(row, constants);
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
    } else {
        sigmoid_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
