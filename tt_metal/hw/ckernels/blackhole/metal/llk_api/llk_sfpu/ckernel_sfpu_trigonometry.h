// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel::sfpu {

static const float PI = 3.1415927f;
static const float PI_2 = 1.5707964f;
static const float PI_4 = 0.7853982f;
static const float FRAC_1_PI = 0.31830987f;

static sfpi_inline sfpi::vFloat sfpu_tan_large(sfpi::vFloat x) {
    const sfpi::vFloat r = 4.0f * sfpi::abs(x) - 5.0f;
    // const sfpi::vFloat y =
    //     ((((((((((((2.1457846f * r + 2.815174f) * r - 3.9035487f) * r - 5.2096696f) * r + 3.658698f) * r
    //     + 4.9457364f) *
    //                r -
    //            0.7137798f) *
    //               r -
    //           1.0665413f) *
    //              r +
    //          1.1057009f) *
    //             r +
    //         1.462757f) *
    //            r +
    //        1.4643353f) *
    //           r +
    //       1.8716435f) *
    //          r +
    //      2.514385f) *
    //         r +
    //     3.0097759f;

    const sfpi::vFloat y = PolynomialEvaluator::eval(
        r,
        3.0097759f,
        2.514385f,
        1.8716435f,
        1.4643353f,
        1.462757f,
        1.1057009f,
        -1.0665413f,
        -0.7137798f,
        4.9457364f,
        3.658698f,
        -5.2096696f,
        -3.9035487f,
        2.815174f,
        2.1457846f);

    return sfpi::setsgn(y, x);
}

template <bool APPROXIMATION_MODE>
static sfpi::vFloat sfpu_tan(sfpi::vFloat x);

template <>
sfpi_inline sfpi::vFloat sfpu_tan<true>(sfpi::vFloat x) {
    const sfpi::vFloat xx = x * x;

    v_if(sfpi::abs(x) <= 1.0f) {
        x = x * PolynomialEvaluator::eval(xx, 1.0000609f, 0.33035427f, 0.1559396f, -0.0031158808f, 0.07407404f);
    }
    v_else { x = sfpu_tan_large(x); }
    v_endif;

    return x;
}

template <>
sfpi_inline sfpi::vFloat sfpu_tan<false>(sfpi::vFloat x) {
    const sfpi::vFloat xx = x * x;

    v_if(sfpi::abs(x) <= 1.0f) {
        // x *= ((((((0.010222361f * xx - 0.015764693f) * xx + 0.02789032f) * xx + 0.012122508f) * xx + 0.05659461f) *
        // xx +
        //        0.1329926f) *
        //           xx +
        //       0.33334994f) *
        //          xx +
        //      0.9999999f;
        x *= PolynomialEvaluator::eval(
            xx,
            0.9999999f,
            0.33334994f,
            0.1329926f,
            0.05659461f,
            0.012122508f,
            0.02789032f,
            -0.015764693f,
            0.010222361f);
    }
    v_else { x = sfpu_tan_large(x); }
    v_endif;

    return x;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_tangent() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0] * FRAC_1_PI;
        v -= sfpi::int32_to_float(sfpi::float_to_int16(v, 0), 0);
        sfpi::dst_reg[0] = sfpu_tan<APPROXIMATION_MODE>(PI * v);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
static sfpi::vFloat sfpu_sinpi(sfpi::vFloat x);

template <>
sfpi_inline sfpi::vFloat sfpu_sinpi<true>(sfpi::vFloat x) {
    sfpi::vFloat xx = x * x;

    return x * PolynomialEvaluator::eval(xx, 0x1.92149p+1f, -0x1.4954d4p+2f, 0x1.29cf02p+1f);
    // return x * ((0x1.29cf02p+1f * xx - 0x1.4954d4p+2f) * xx + 0x1.92149p+1f);
}

template <>
sfpi_inline sfpi::vFloat sfpu_sinpi<false>(sfpi::vFloat x) {
    sfpi::vFloat xx = x * x;

    // return x *
    //    ((((0x1.406628p-4f * xx - 0x9.93f86p-4f) * xx + 0x2.8cd64p+0f) * xx - 0x5.2aef6p+0f) * xx + 0x3.243f6cp+0f);

    return x *
           PolynomialEvaluator::eval(xx, 0x3.243f6cp+0f, -0x5.2aef6p+0f, 0x2.8cd64p+0f, -0x9.93f86p-4f, 0x1.406628p-4f);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sine() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0] * FRAC_1_PI;
        sfpi::vInt whole_v = sfpi::float_to_int16(v, 0);
        v -= sfpi::int32_to_float(whole_v, 0);
        v = sfpu_sinpi<APPROXIMATION_MODE>(v);

        v_if(whole_v & 1) { v = -v; }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_cosine() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0] * FRAC_1_PI + 0.5f;
        sfpi::vInt whole_v = sfpi::float_to_int16(v, 0);
        v -= sfpi::int32_to_float(whole_v, 0);
        v = sfpu_sinpi<APPROXIMATION_MODE>(v);

        v_if(whole_v & 1) { v = -v; }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sfpu_trig() {
    if constexpr (operation == SfpuType::sine) {
        calculate_sine<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::cosine) {
        calculate_cosine<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::tan) {
        calculate_tangent<APPROXIMATION_MODE, ITERATIONS>();
    }
}

template <bool is_fp32_dest_acc_en, bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_atan(sfpi::vFloat val) {
    sfpi::vFloat t0 = sfpi::abs(val);
    v_if(t0 > 1) { t0 = sfpu_reciprocal<false>(t0); }
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
            -0.3333314528f,
            0.1999355085f,
            -0.1420889944f,
            0.1065626393f,
            -0.07528964f,
            0.0429096138f,
            -0.0161657367f,
            0.0028662257f);
    }

    t1 = t1 * t0;

    v_if(sfpi::abs(val) > 1) { t1 = 1.570796327f - t1; }
    v_endif;

    return sfpi::setsgn(t1, val);
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
sfpi_inline sfpi::vFloat sfpu_asine_maclaurin_series(sfpi::vFloat val) {
    // input for [-1:1]
    // Mclaurin series
    // arcsin(x) = x + [(1/2) *x^3/3] + [(1 * 3) / (2 * 4) * x^5 / 5] + [(1 * 3 * 5) / (2 * 4 * 6) * x^7 / 7 ] + ...
    // arcsin(x) ≈ x + (1/6) * x^3 + (3/40) * x^5 + (5/112) * x^7 + (35/1152) * x^9 + (63/2816) * x^11

    sfpi::vFloat tmp = val;
    sfpi::vFloat x = val;
    sfpi::vFloat xx = tmp * tmp;

    sfpi::vFloat output = x * PolynomialEvaluator::eval(
                                  xx,
                                  sfpi::vConst1,
                                  0.166666666f,  // (1/6) * x^3
                                  0.075f,        // (3/40) * x^5
                                  0.044642857f,  // (5/112) * x^7
                                  0.03038194f,   // (35/1152) * x^9
                                  0.02237216f    // (63/2816) * x^11
                              );

    return output;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_mode, int ITERATIONS>
inline void calculate_asin() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result;

        v_if(v < sfpi::vConstNeg1 || v > sfpi::vConst1) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_else { result = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v); }
        v_endif;

        if constexpr (!is_fp32_dest_acc_mode) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_mode, int ITERATIONS>
inline void calculate_acos() {
    // SFPU microcode
    // acos = (pi/2 - asin)
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result;

        v_if(v < sfpi::vConstNeg1 || v > sfpi::vConst1) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_else { result = PI_2 - sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v); }
        v_endif;

        if constexpr (!is_fp32_dest_acc_mode) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// cosh = (exp(x) + exp(-x)) / 2
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cosh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result =
            (_sfpu_exp_improved_<is_fp32_dest_acc_en>(v) + _sfpu_exp_improved_<is_fp32_dest_acc_en>(-v)) * 0.5f;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// sinh = (exp(x) - exp(-x)) / 2
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_sinh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result =
            (_sfpu_exp_improved_<is_fp32_dest_acc_en>(v) - _sfpu_exp_improved_<is_fp32_dest_acc_en>(-v)) * 0.5f;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
        kj
    }
}

template <bool APPROXIMATION_MODE>
void atan_init() {
    // Initialisation for use of sfpu_reciprocal<false>.
    sfpu_reciprocal_init<false>();
}

template <bool APPROXIMATION_MODE>
void init_hyperbolic_trig() {
    _init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>();
}

}  // namespace ckernel::sfpu
