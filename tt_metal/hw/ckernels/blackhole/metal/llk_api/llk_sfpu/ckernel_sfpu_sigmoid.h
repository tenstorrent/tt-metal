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

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
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

// c1, c2 of exp_21f for the fraction in [0, 1) rather than for the fraction scaled by 2**23.
constexpr float EXP_21F_BF16_C1_FRAC = EXP_21F_BF16_C1 * 0x1p23f;
constexpr float EXP_21F_BF16_C2_FRAC = EXP_21F_BF16_C2 * 0x1p46f;

// -(1/ln2), which _sfpu_sigmoid_bf16_rows_'s caller loads once before its row loop, and the clamp's upper bound.
struct SigmoidBf16Constants {
    sfpi::vFloat neg_one_ln2 = -1.4426950216293334961f;
    static constexpr float t_max = 254.0f;
};

// The BF16-grade sigmoid of _sfpu_sigmoid_<false> (exp_21f of -x, then one Newton step of the reciprocal of
// 1 + exp(-x)) of N = 1 or 2 independent rows, step by step across the rows so that each row fills the other's MAD
// latency; bit-identical to _sfpu_sigmoid_<false> for every input (test_sfpu_bit_identity.py). exp_21f's exponent is
// clamped to [0, 254], so exp(-x) stays finite and the reciprocal needs no NaN guard. Two rows use all eight lregs:
// the caller loads -(1/ln2) once before its row loop, the other constants are loaded where used. With
// constants_in_cregs the clamp bound t_max and c2 come from vConstFloatPrgm1 and vConstFloatPrgm2 (silu_init).
// Requires vConstFloatPrgm0 = 2.0.
template <bool constants_in_cregs, int N>
sfpi_inline void _sfpu_sigmoid_bf16_rows_(sfpi::vFloat (&x)[N], const SigmoidBf16Constants& constants) {
    static_assert(N == 1 || N == 2, "two rows use the eight lregs");
    using C = SigmoidBf16Constants;
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
            t[i] = sfpi::min(t[i], sfpi::vFloat(C::t_max));
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
    c = EXP_21F_BF16_C1_FRAC;
#pragma GCC unroll 2
    for (int i = 0; i < N; i++) {
        if constexpr (constants_in_cregs) {
            t[i] = f[i] * sfpi::vConstFloatPrgm2 + c;
        } else {
            t[i] = f[i] * EXP_21F_BF16_C2_FRAC + c;
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

// sigmoid(x) = 1 / (1 + exp(-x)) of N = 1 or 2 independent rows, step by step across the rows so that each row fills
// the other's latency. Bit-identical to _sfpu_sigmoid_<true> (exp_fp32_accurate of -x, then two Newton steps of the
// reciprocal of 1 + exp(-x)) for every non-NaN x (test_sfpu_bit_identity.py). |x| is clamped to 88 first: beyond it
// the result is already exactly 0 or 1, and within it exp(-x) and 1 + exp(-x) stay finite and normal, so neither the
// exponential nor the Newton steps need a guard; 2**j is built from j's bits. A +NaN x gives 1 and a -NaN x NaN
// (the unclamped form gives 0 for either). Two rows fit seven lregs: the caller loads -(1/ln2) once into the eighth,
// the other constants are loaded where used. Requires vConstFloatPrgm0 = 2.0 (sigmoid_init). With
// constants_in_cregs the clamp bound 88 and the 2**j bias come from vConstFloatPrgm1 and vConstFloatPrgm2
// (sigmoid_init) rather than SFPLOADIs.
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
        // The FP32 path's constants (_sfpu_sigmoid_fp32_rows_<true>); the BF16 path uses neither. sigmoid_tile must
        // follow sigmoid_init with no other SFPU init in between: the FP32 path reads vConstFloatPrgm1/2 as well as
        // Prgm0, and an init that keeps Prgm0 = 2.0 (silu_tile_init, for one) can still overwrite Prgm1/2.
        sfpi::vConstFloatPrgm1 = SigmoidFp32Constants::clamp;
        sfpi::vConstFloatPrgm2 = Converter::as_float(SigmoidFp32Constants::exp2_bias_bits);
    } else {
        sigmoid_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
