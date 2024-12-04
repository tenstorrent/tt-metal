// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {

namespace sfpu {

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

#define POLYVAL6(coef5, coef4, coef3, coef2, coef1, coef0, t4) \
    (t4 * (t4 * (t4 * (t4 * (coef5 * t4 + coef4) + coef3) + coef2) + coef1) + coef0)

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_atan_maclaurin_series(vFloat val) {
    v_if(1 > sfpi::abs(val)) { dst_reg[0] = sfpi::abs(val); }
    v_else { dst_reg[0] = sfpu_reciprocal(sfpi::abs(val)); }
    v_endif;

    vFloat t1 = dst_reg[0] * dst_reg[0];

    t1 = POLYVAL6(-0.013480470f, 0.057477314f, -0.121239071f, 0.195635925f, -0.332994597f, 0.999995630f, t1);

    t1 = t1 * dst_reg[0];

    v_if(sfpi::abs(val) > 1) { t1 = 1.570796327f - t1; }
    v_endif;

    v_if(val < 0) { t1 = -t1; }
    v_endif;

    return t1;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atan() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        val = sfpu_atan_maclaurin_series<APPROXIMATION_MODE>(val);
        dst_reg[0] = val;
        dst_reg++;
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
        v = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v);
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acos() {
    // SFPU microcode
    // acos = (pi/2 - asin)
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v);
        v = PI_2 - v;
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void atan_init() {
    vConstFloatPrgm0 = 1.442695f;  // ln2_recip
    vConstFloatPrgm1 = 2.0f;
}

}  // namespace sfpu
}  // namespace ckernel
