// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

static sfpi_inline vFloat sfpu_tan_large(vFloat x) {
    const vFloat r = 4.0f * sfpi::abs(x) - 5.0f;
    const vFloat y =
        ((((((((((((2.1457846f * r + 2.815174f) * r - 3.9035487f) * r - 5.2096696f) * r + 3.658698f) * r + 4.9457364f) *
                   r -
               0.7137798f) *
                  r -
              1.0665413f) *
                 r +
             1.1057009f) *
                r +
            1.462757f) *
               r +
           1.4643353f) *
              r +
          1.8716435f) *
             r +
         2.514385f) *
            r +
        3.0097759f;
    return setsgn(y, x);
}

template <bool APPROXIMATION_MODE>
static vFloat sfpu_tan(vFloat x);

template <>
sfpi_inline vFloat sfpu_tan<true>(vFloat x) {
    const vFloat xx = x * x;

    v_if(sfpi::abs(x) <= 1.0f) {
        x *= (((0.07407404f * xx - 0.0031158808f) * xx + 0.1559396f) * xx + 0.33035427) * xx + 1.0000609f;
    }
    v_else { x = sfpu_tan_large(x); }
    v_endif;

    return x;
}

template <>
sfpi_inline vFloat sfpu_tan<false>(vFloat x) {
    const vFloat xx = x * x;

    v_if(sfpi::abs(x) <= 1.0f) {
        x *= ((((((0.010222361f * xx - 0.015764693f) * xx + 0.02789032f) * xx + 0.012122508f) * xx + 0.05659461f) * xx +
               0.1329926f) *
                  xx +
              0.33334994f) *
                 xx +
             0.9999999f;
    }
    v_else { x = sfpu_tan_large(x); }
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

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sine() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0] * FRAC_1_PI;
        vInt whole_v = float_to_int16(v, 0);
        v -= int32_to_float(whole_v, 0);
        v = sfpu_sinpi<APPROXIMATION_MODE>(v);

        v_if(whole_v & 1) { v = -v; }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_cosine() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0] * FRAC_1_PI + 0.5f;
        vInt whole_v = float_to_int16(v, 0);
        v -= int32_to_float(whole_v, 0);
        v = sfpu_sinpi<APPROXIMATION_MODE>(v);

        v_if(whole_v & 1) { v = -v; }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
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
