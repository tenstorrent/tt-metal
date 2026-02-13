// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel::sfpu {

static const float PI = 3.1415927f;
static const float PI_2 = 1.5707964f;
static const float PI_4 = 0.7853982f;
static const float FRAC_1_PI = 0.31830987f;

static const float PI_2_HI = 0x1.921fb4p+0f;
static const float PI_2_LO = 0x1.4442d2p-24f;

static sfpi_inline vFloat _tan_(vFloat s) {
    vFloat r;

    r = 4.38117981e-3f;          // 0x1.1f2000p-8
    r = r * s + 8.94600598e-5f;  // 0x1.773902p-14
    r = r * s + 1.08341556e-2f;  // 0x1.63037cp-7
    r = r * s + 2.12811474e-2f;  // 0x1.5cab9ap-6
    r = r * s + 5.40602170e-2f;  // 0x1.badc7ep-5
    r = r * s + 1.33326918e-1f;  // 0x1.110db4p-3
    r = r * s + 3.33333433e-1f;  // 0x1.110db4p-3
    r = r * s;

    return r;
}

template <bool APPROXIMATION_MODE>
static vFloat sfpu_tan(vFloat x);

template <>
sfpi_inline vFloat sfpu_tan<true>(vFloat x) {
    return x;
}

template <>
sfpi_inline vFloat sfpu_tan<false>(vFloat x) {
    const vFloat s = x * x;

    vFloat t = _tan_(s);
    vFloat r = t * x + x;
    v_if(i & 1) {
        s = x - r;
        s = t * x + s;
        t = sfpi::approx_recip(-r);
        r = r * t + sfpi::vConst1;
        r = s * t + r;
        r = r * t + t;
    }
    v_endif;

    return x;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_tangent() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0] * FRAC_1_PI;
        v -= int32_to_float(float_to_int16(v, 0), 0);
        dst_reg[0] = sfpu_tan<APPROXIMATION_MODE>(PI * v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
static vFloat sfpu_sinpi(vFloat x);

template <>
sfpi_inline vFloat sfpu_sinpi<true>(vFloat x) {
    vFloat xx = x * x;

    return x * ((0x1.29cf02p+1f * xx - 0x1.4954d4p+2f) * xx + 0x1.92149p+1f);
}

template <>
sfpi_inline vFloat sfpu_sinpi<false>(vFloat x) {
    vFloat xx = x * x;

    return x *
           ((((0x1.406628p-4f * xx - 0x9.93f86p-4f) * xx + 0x2.8cd64p+0f) * xx - 0x5.2aef6p+0f) * xx + 0x3.243f6cp+0f);
}

template <bool IS_COSINE, bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sincos_internal() {
    // 1. Reduce argument using a four-stage Cody-Waite reduction to the interval [-pi/2, pi/2].
    // 2. In the case of cos(x), use cos(x) = sin(pi / 2 - abs(x)), using a two-part subtraction to avoid precision
    // loss.
    // 3. Now use a polynomial sin(x) = x + x^3 (C0 + x^2 (C1 + x^2 (C2 + x^2 C3))) on [0, pi/2].

    // Constants for four-stage Cody-Waite reduction with -PI = P0 + P1 + vConstFloatPrgm0 + vConstFloatPrgm1
    const float P0 = -0x1.92p+1f;               // representable as bf16
    const float P1 = -0x1.fbp-11f;              // representable as fp16
    sfpi::vConstFloatPrgm0 = -0x1.51p-21f;      // requires fp32
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-33f;  // requires fp32

    sfpi::vConstFloatPrgm2 = FRAC_1_PI;

    // Constants for sin(a) = a + a^3 (C0 + a^2 (C1 + a^2 (C2 + a^2 C3))) on [0, pi/2].
    vFloat C3 = 0x1.5dc908p-19f;
    vFloat C2 = -0x1.9f70fp-13f;
    vFloat C1 = 0x1.110edap-7f;
    vFloat C0 = -0x1.55554cp-3f;

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        // Workaround for SFPI's insistence on generating SFPADDI+SFPMUL instead of SFPLOADI+SFPMAD here.
        vFloat rounding_bias;
        rounding_bias.get() = __builtin_rvtt_sfpxloadi(0, 0x4b40);  // 1.5*2^23
        __rvtt_vec_t inv_pi = __builtin_rvtt_sfpreadlreg(sfpi::vConstFloatPrgm2.get());

        // Compute j = round(v / PI).
        // First, j = v * (1 / PI) + 1.5*2^23 shifts the mantissa bits to give round-to-nearest-even.
        vFloat j;
        j.get() = __builtin_rvtt_bh_sfpmad(v.get(), inv_pi, rounding_bias.get(), SFPMAD_MOD1_OFFSET_NONE);

        // At this point, the mantissa bits of j contain the integer.
        // Store for later; the LSB determines the sign of the result.
        vInt q = reinterpret<vInt>(j);
        // Shift mantissa bits back; j is now round(v / PI) in fp32.
        j = j - rounding_bias;

        // Four-stage Cody-Waite reduction; a = v + j * -PI.
        // P0 representable as bf16; generates a single SFPLOADI, filling NOP slot from previous SFPADDI.
        vFloat a = v + j * P0;
        // P1 representable as fp16; generates a single SFPLOADI, filling NOP slot from previous SFPMAD.
        a = a + j * P1;
        a = a + j * sfpi::vConstFloatPrgm0;
        a = a + j * sfpi::vConstFloatPrgm1;

        // If we're computing cosine, we want cos(a) = sin(pi/2 - abs(a)).
        // Use two-part subtraction to avoid precision loss.
        if constexpr (IS_COSINE) {
            a = sfpi::abs(a);
            a = PI_2 - a;
            a = a + PI_2_LO;
        }

        q <<= 31;
        vFloat s = a * a;
        a = reinterpret<vFloat>(reinterpret<vInt>(a) ^ q);
        vFloat r = C3 * s + C2;
        r = r * s + C1;
        vFloat c = a * s;
        r = r * s + C0;
        r = r * c + a;

        dst_reg[0] = r;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sine() {
    calculate_sincos_internal<false, APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_cosine() {
    // 1. Reduce argument using a four-stage Cody-Waite reduction to the interval [-pi/2, pi/2].
    // 2. In the case of cos(x), use cos(x) = sin(pi / 2 + x).
    // 3. Now use a polynomial sin(x) = x + x^3 (C0 + x^2 (C1 + x^2 (C2 + x^2 C3))) on [0, pi/2].

    // Constants for four-stage Cody-Waite reduction with -PI/2 = P0 + P1 + vConstFloatPrgm0 + vConstFloatPrgm1
    const float P0 = -0x1.92p+0f;               // representable as bf16
    const float P1 = -0x1.fbp-12f;              // representable as fp16
    sfpi::vConstFloatPrgm0 = -0x1.51p-22f;      // requires fp32
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-34f;  // requires fp32

    sfpi::vConstFloatPrgm2 = FRAC_1_PI;

    // Constants for sin(a) = a + a^3 (C0 + a^2 (C1 + a^2 (C2 + a^2 C3))) on [0, pi/2].
    vFloat C3 = 0x1.5dc908p-19f;
    vFloat C2 = -0x1.9f70fp-13f;
    vFloat C1 = 0x1.110edap-7f;
    vFloat C0 = -0x1.55554cp-3f;

    const float ROUNDING_BIAS = 12582912.0f;
    const float NEG_ROUNDING_BIAS = -12582912.0f;

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        // Workaround for SFPI's insistence on generating SFPADDI+SFPMUL instead of SFPLOADI+SFPMAD here.
        vFloat half;
        half.get() = __builtin_rvtt_sfpxloadi(0, 0x3f00);  // 0.5
        __rvtt_vec_t inv_pi = __builtin_rvtt_sfpreadlreg(sfpi::vConstFloatPrgm2.get());
        __rvtt_vec_t one = __builtin_rvtt_sfpreadlreg(sfpi::vConst1.get());
        __rvtt_vec_t neg_one = __builtin_rvtt_sfpreadlreg(sfpi::vConstNeg1.get());

        // Compute j = round(v / PI).
        // First, j = v * (2 / PI) + 1.5*2^23 shifts the mantissa bits to give round-to-nearest-even.
        vFloat j;
        j.get() = __builtin_rvtt_bh_sfpmad(v.get(), inv_pi, half.get(), SFPMAD_MOD1_OFFSET_NONE);

        // vFloat rounding_bias;
        // rounding_bias.get() = __builtin_rvtt_sfpxloadi(0, 0x4b40);  // 1.5*2^23
        // j.get() = __builtin_rvtt_bh_sfpmad(v.get(), one, rounding_bias.get(), SFPMAD_MOD1_OFFSET_NONE);

        j = j + ROUNDING_BIAS;

        // At this point, the mantissa bits of j contain the integer.
        // Store for later; the LSB determines the sign of the result.
        vInt q = reinterpret<vInt>(j);

        j = j + NEG_ROUNDING_BIAS;

        vFloat two;
        two.get() = __builtin_rvtt_sfpxloadi(0, 0x4000);  // 2.0
        j.get() = __builtin_rvtt_bh_sfpmad(j.get(), two.get(), neg_one, SFPMAD_MOD1_OFFSET_NONE);

        // Four-stage Cody-Waite reduction; a = v + j * -PI / 2.
        // P0 representable as bf16; generates a single SFPLOADI, filling NOP slot from previous SFPADDI.
        vFloat a = v + j * P0;
        // P1 representable as fp16; generates a single SFPLOADI, filling NOP slot from previous SFPMAD.
        a = a + j * P1;
        a = a + j * sfpi::vConstFloatPrgm0;
        a = a + j * sfpi::vConstFloatPrgm1;

        q <<= 31;
        vFloat s = a * a;
        a = reinterpret<vFloat>(reinterpret<vInt>(a) ^ q);
        vFloat r = C3 * s + C2;
        r = r * s + C1;
        vFloat c = a * s;
        r = r * s + C0;
        r = r * c + a;

        dst_reg[0] = r;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat sfpu_atan(sfpi::vFloat val) {
    sfpi::vFloat t0 = sfpi::abs(val);
    sfpi::vFloat result = sfpi::vConst0;

    // If input is NaN then output must be NaN as well
    sfpi::vInt exponent = sfpi::exexp_nodebias(val);
    sfpi::vInt mantissa = sfpi::exman9(val);
    v_if(exponent == 255 && mantissa != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_else {
        sfpi::vFloat absval_minus_1 = t0 - sfpi::vConst1;

        v_if(absval_minus_1 > 0.0f) { t0 = sfpu_reciprocal<false>(t0); }
        v_endif;

        sfpi::vFloat t1 = t0 * t0;

        if constexpr (!is_fp32_dest_acc_en) {
            // Found using Sollya
            // > fpminimax(atan(x), [|1,3,5,7|], [|single...|], [2^(-40); 1], relative);
            t1 = PolynomialEvaluator::eval(
                t1,
                0.999787867069244384765625f,
                -0.325808584690093994140625f,
                0.1555790007114410400390625f,
                -4.4326744973659515380859375e-2f);
        } else {
            // Found using Sollya
            // > fpminimax(atan(x), [|1,3,5,7,9,11,13,15,17|], [|single...|], [2^(-40); 1], relative);
            t1 = PolynomialEvaluator::eval(
                t1,
                sfpi::vConst1,
                -0.3333314359188079833984375f,
                0.19993579387664794921875f,
                -0.14209578931331634521484375f,
                0.1066047251224517822265625f,
                -7.5408883392810821533203125e-2f,
                4.3082617223262786865234375e-2f,
                -1.62907354533672332763671875e-2f,
                2.90188402868807315826416015625e-3f);
        }

        t1 = t1 * t0;

        v_if(absval_minus_1 > 0.0f) { t1 = PI_2 - t1; }
        v_endif;

        result = sfpi::setsgn(t1, val);
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_atan() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = sfpu_atan<APPROXIMATION_MODE, is_fp32_dest_acc_en>(in);

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_asine_maclaurin_series(vFloat val) {
    // input for [-1:1]
    // Mclauren series
    // arcsin(x) = x + [(1/2) *x^3/3] + [(1 * 3) / (2 * 4) * x^5 / 5] + [(1 * 3 * 5) / (2 * 4 * 6) * x^7 / 7 ] + ...
    // arcsin(x) ≈ x + (1/6) * x^3 + (3/40) * x^5 + (5/112) * x^7 + (35/1152) * x^9 + (63/2816) * x^11a

    vFloat tmp = val;
    vFloat val_square = val * val;
    // x
    vFloat output = tmp;
    // (1/6) * x^3
    tmp = tmp * val_square;
    output += 0.166666666 * tmp;
    // (3/40) * x^5
    tmp = tmp * val_square;
    output += 0.075 * tmp;

    //(5/112) * x^7
    tmp = tmp * val_square;
    output += 0.044642857 * tmp;

    // (35/1152) *x^9
    tmp = tmp * val_square;
    output += 0.03038194 * tmp;

    //(63/2816) * x^11
    tmp = tmp * val_square;
    output += 0.02237216 * tmp;

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asin() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v < vConstNeg1 || v > vConst1) { dst_reg[0] = std::numeric_limits<float>::quiet_NaN(); }
        v_else { dst_reg[0] = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v); }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acos() {
    // SFPU microcode
    // acos = (pi/2 - asin)
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v < vConstNeg1 || v > vConst1) { dst_reg[0] = std::numeric_limits<float>::quiet_NaN(); }
        v_else { dst_reg[0] = PI_2 - sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v); }
        v_endif;
        dst_reg++;
    }
}

// cosh = (exp(x) + exp(-x)) / 2
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cosh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        vFloat result = (_sfpu_exp_21f_<is_fp32_dest_acc_en>(v) + _sfpu_exp_21f_<is_fp32_dest_acc_en>(-v)) * 0.5f;
        dst_reg[0] = result;
        dst_reg++;
    }
}

// sinh = (exp(x) - exp(-x)) / 2
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_sinh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        vFloat result = (_sfpu_exp_21f_<is_fp32_dest_acc_en>(v) - _sfpu_exp_21f_<is_fp32_dest_acc_en>(-v)) * 0.5f;
        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_hyperbolic_trig() {
    _init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>();
}

template <bool APPROXIMATION_MODE>
void atan_init() {
    // Initialisation for use of sfpu_reciprocal<false>.
    sfpu_reciprocal_init<false>();
}

}  // namespace ckernel::sfpu
