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

#define PI (3.14159265358979323846)
#define PI_2 (1.570796326794)

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_tangent_maclaurin_series(vFloat val) {
    // Mclauren series
    // tan(x) = x + (x^3)/3 + (2x^5)/15 + (17x^7)/315 + (62x^9)/2835 + (1382x^11)/155925 + (21844x^13)/6081075 + ...

    vFloat tmp = val;
    vFloat val_square = val * val;

    // x
    vFloat output = tmp;
    // x^3/3
    tmp = tmp * val_square;
    output += 0.3333333333333333 * tmp;
    // (2x^5)/15
    tmp = tmp * val_square;
    output += 0.13333333333333333 * tmp;

    //(17x^7)/315
    tmp = tmp * val_square;
    output += 0.05396825396825397 * tmp;

    //(62x^9)/2835
    tmp = tmp * val_square;
    output += 0.021869488536155203 * tmp;

    // (1382x^11)/155925
    tmp = tmp * val_square;
    output += 0.008863235529902197 * tmp;

    // (21844x^13)/6081075
    tmp = tmp * val_square;
    output += 0.003592128036572481 * tmp;

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_tangent() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        // Periodic, Range Reduction: To cover more input range
        v_if(v > PI_2) { v = v - PI; }
        v_elseif(v < -PI_2) { v = v + PI; }
        v_else { v = v; }
        v_endif;

        v = sfpu_tangent_maclaurin_series<APPROXIMATION_MODE>(v);
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_sine_maclaurin_series(vFloat val) {
    // Good for [-pi:pi]
    // Mclauren series = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    vFloat tmp = val;
    // x
    vFloat output = tmp;
    // x^3/3!
    tmp = tmp * val * val;
    output += -0.166666666 * tmp;
    // x^5/5!
    tmp = tmp * val * val;
    output += 0.0083333333 * tmp;
    // x^7/7!
    tmp = tmp * val * val;
    output += -0.0001984126 * tmp;

    // x^9/9!
    tmp = tmp * val * val;
    output += 0.0000027557 * tmp;

    // x^11/11!
    tmp = tmp * val * val;
    output += -0.00000002505 * tmp;

    if constexpr (not APPROXIMATION_MODE) {
        // x^11/11!
        tmp = tmp * val * val;
        output += -0.00000002505 * tmp;

        // x^13/13!
        tmp = tmp * val * val;
        output += 1.6059043836821613e-10 * (tmp);
    }

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_cosine_maclaurin_series(vFloat val) {
    // Good for [-pi:pi]
    // Mclauren series = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12!
    // 1
    vFloat output = 1.0f;
    // x^2/2!
    vFloat tmp = val * val;
    output += -0.5 * tmp;
    // x^4/4!
    tmp = tmp * val * val;
    output += 0.0416666666 * tmp;
    // x^6/6!
    tmp = tmp * val * val;
    output += -0.0013888888 * tmp;

    // x^8/8!
    tmp = tmp * val * val;
    output += 0.0000248015 * tmp;

    // x^10/10!
    tmp = tmp * val * val;
    output += -0.0000002755 * tmp;

    if constexpr (not APPROXIMATION_MODE) {
        // x^12/12!
        tmp = tmp * val * val;
        output += 2.08767569878681e-9 * tmp;

        // x^14/14!
        tmp = tmp * val * val;
        output += -1.1470745597729725e-11 * tmp;
    }

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sine() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v = 0.318309886183791f * v;  // *1/pi to get number of pi rads.
        vInt whole_v = float_to_int16(v);
        vFloat whole_v_float = int32_to_float(whole_v, 0);
        v = v - whole_v_float;
        v *= 3.141592653589793f;  // fractional * pi to get it in [-pi:pi]
        v = sfpu_sine_maclaurin_series<APPROXIMATION_MODE>(v);
        whole_v = whole_v & 0x1;
        v_if(whole_v != 0) {
            // odd so flip the sign
            v *= -1;
        }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_cosine() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v = 0.318309886183791f * v;  // *1/pi to get number of pi rads.
        vInt whole_v = float_to_int16(v);
        vFloat whole_v_float = int32_to_float(whole_v, 0);
        v = v - whole_v_float;
        v *= 3.141592653589793f;  // fractional * pi to get it in [-pi:pi]
        v = sfpu_cosine_maclaurin_series<APPROXIMATION_MODE>(v);
        whole_v = whole_v & 0x1;
        v_if(whole_v != 0) {
            // odd so flip the sign
            v *= -1;
        }
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
