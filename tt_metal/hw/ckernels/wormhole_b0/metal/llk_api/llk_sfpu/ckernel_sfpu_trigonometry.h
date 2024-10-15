// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_recip.h"

using namespace sfpi;

namespace ckernel {

namespace sfpu {

static const float PI = 3.1415927f;
static const float PI_2 = 1.5707964f;
static const float FRAC_1_PI = 0.31830987f;

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_tangent_maclaurin_series(vFloat val)
{
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
inline void calculate_tangent()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        //Periodic, Range Reduction: To cover more input range
        v_if(v > PI_2){
            v = v - PI;
        }v_elseif(v < -PI_2){
            v = v + PI;
        }v_else{
            v = v;
        }v_endif;

        v = sfpu_tangent_maclaurin_series<APPROXIMATION_MODE>(v);
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
static vFloat sfpu_sinpi(vFloat x);

template <>
sfpi_inline vFloat sfpu_sinpi<true>(vFloat x)
{
    vFloat xx = x * x;

    return x * ((0x1.29cf02p+1f
        * xx - 0x1.4954d4p+2f)
        * xx + 0x1.92149p+1f);
}

template <>
sfpi_inline vFloat sfpu_sinpi<false>(vFloat x)
{
    vFloat xx = x * x;

    return x * ((((0x1.406628p-4f
        * xx - 0x9.93f86p-4f)
        * xx + 0x2.8cd64p+0f)
        * xx - 0x5.2aef6p+0f)
        * xx + 0x3.243f6cp+0f);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sine()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0] * FRAC_1_PI;
        vInt whole_v = float_to_int16(v);
        v -= int32_to_float(whole_v, 0);
        v = sfpu_sinpi<APPROXIMATION_MODE>(v);

        v_if (whole_v & 1) {
            v = -v;
        }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_cosine()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0] * FRAC_1_PI + 0.5f;
        vInt whole_v = float_to_int16(v);
        v -= int32_to_float(whole_v, 0);
        v = sfpu_sinpi<APPROXIMATION_MODE>(v);

        v_if (whole_v & 1) {
            v = -v;
        }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS=8>
inline void calculate_sfpu_trig() {
    if constexpr (operation == SfpuType::sine) {
        calculate_sine<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::cosine) {
        calculate_cosine<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::tan) {
        calculate_tangent<APPROXIMATION_MODE, ITERATIONS>();
    }
}

#define POLYVAL6(coef5, coef4, coef3, coef2, coef1, coef0, t4)  (t4 * (t4 * (t4 * (t4 * (coef5 * t4 + coef4) + coef3) + coef2) + coef1) + coef0)

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_atan_maclaurin_series(vFloat val)
{
    v_if(1 > sfpi::abs(val)){
        dst_reg[0] = sfpi::abs(val)  ;
    }
    v_else{
        dst_reg[0] =  sfpu_reciprocal(sfpi::abs(val));
    }
    v_endif;

    vFloat t1 = dst_reg[0] * dst_reg[0];

    t1 = POLYVAL6(-0.013480470f, 0.057477314f, -0.121239071f, 0.195635925f, -0.332994597f, 0.999995630f, t1);

    t1 = t1 * dst_reg[0];

    v_if (sfpi::abs(val) > 1){
        t1 = 1.570796327f - t1;
    }
    v_endif;

    v_if(val < 0 ){
        t1 = -t1;
    }
    v_endif;

    return t1;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atan()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        val = sfpu_atan_maclaurin_series<APPROXIMATION_MODE>(val);
        dst_reg[0] = val;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_asine_maclaurin_series(vFloat val)
{
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
    output +=  0.075 * tmp;

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
inline void calculate_asin()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        v = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v);
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_acos()
{
    // SFPU microcode
    // acos = (pi/2 - asin)
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        v = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v);
        v = PI_2 - v;
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void atan_init() {
    vConstFloatPrgm0 = 1.442695f; // ln2_recip
    vConstFloatPrgm1 = 2.0f;
}

}  // namespace sfpu
}  // namespace ckernel
