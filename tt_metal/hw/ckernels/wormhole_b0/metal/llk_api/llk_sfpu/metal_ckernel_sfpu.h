// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

#include "ckernel_sfpu_cdf.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS, int RECIPROCAL_ITERATIONS>
inline void calculate_rsqrt()
{

    for (int d = 0; d < ITERATIONS; d++)
    {

        vFloat in = dst_reg[0];
        v_if(dst_reg[0] == 0.0f){
            dst_reg[0] = std::numeric_limits<float>::infinity();
        }v_else{
            vFloat result = 1.0f;
            v_if(dst_reg[0] > 1.0f){
                result = sfpu_reciprocal(in);
            }v_endif;

            for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
            {
                // y = y * (1.5 - 0.5 * x * y * y) Newton's method iteration.
                result = result * (1.5F - 0.5F  * dst_reg[0] * result * result);
            }
            dst_reg[0] = result;
        }v_endif;

        dst_reg++;

    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sigmoid_appx()
{
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        dst_reg[0] = lut(val, l0, l1, l2) + 0.5f;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

// TODO: Implement using bitwise comparision
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit()
{

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        v_if (val <= -0.0f) {
            val = 1.0f;
        } v_elseif (val >= 0.0f) {
            val = 0.0f;
        }
        v_endif;
        dst_reg[0] = val;

       dst_reg++;
    }

}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_tanh()
{
    // SFPU microcode
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        val = lut(val, l0, l1, l2);
        dst_reg[0] = val;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardtanh(uint param0, uint param1, uint param2)
{
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    vFloat p0 = s2vFloat16(param0);
    vFloat p1 = s2vFloat16(param1);
    vFloat p2 = s2vFloat16(param2);
    // SFPU microcode
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        val += p0;// 12 bits
        v_if (val < 0.0f) {
            val = 0.0f;
        }
        v_endif;

        val += p1;// 12 bits
        v_if (val >= 0.0f) {
            val = 0.0f;
        }
        v_endif;

        val += p2;// 12 bits

        dst_reg[0] = val;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int WITH_PRECOMPUTED_TANH, int ITERATIONS>
inline void calculate_tanh_derivative()
{
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    // tanh'(x) = 1 - (tanh(x))^2
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        if constexpr (!WITH_PRECOMPUTED_TANH) {
            val = lut(val, l0, l1, l2);
        }

        val = val * (-val) + vConst1;
        dst_reg[0] = val;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_dropout(uint prob, uint scale)
{
    // SFPU microcode

    vUInt rand = l_reg[LRegs::LReg3];
    vUInt probv = prob;

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        ////////////////////////
        // Scale samples
        ///////////////////////
        dst_reg[0] = dst_reg[0] * s2vFloat16b(scale);

        ////////////////////////
        // Drop samples
        ///////////////////////
        v_if (rand < probv) {
            dst_reg[0] = vConst0;
        }
        v_endif;

        ////////////////////////
        // 16-bit PRNG update
        ///////////////////////
        vUInt lfsr = vConstIntPrgm1;
        vUInt tmp = lfsr & rand;
        rand = rand >> 1;
        v_if (tmp != 0) {
            vUInt mask = vConstIntPrgm0;
            rand ^= mask;
        }
        v_endif;

        dst_reg++;
    }

    l_reg[LRegs::LReg3] = rand;
}

template <bool APPROXIMATION_MODE,int ITERATIONS>
inline void calculate_power_iterative(const uint exponent)
{
    #pragma GCC unroll 8
    for (int d = 0; d < 8; d++)
    {
        vFloat in = dst_reg[0];
        vFloat result = 1.0f;
        for (uint i = 0; i < exponent; i++) {
            result *= in;
        }
	dst_reg[0]=result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_square()
{
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat in = dst_reg[0];
        vFloat result = in * in;

        dst_reg[0] = result;

        dst_reg++;
    }
}

template <bool HAS_BASE_SCALING>
sfpi_inline void calculate_log_body(const uint log_base_scale_factor)
{
    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    vFloat in = dst_reg[0];
    vFloat x = setexp(in, 127);    // set exp to exp bias (put in range of 1-2)

    // XXXXXX ask Namal? if we can derive the coefficients below to higher precision
    ////////////////////////////
    // Calculate Cheby Approximation using Horner Form Multiplication: 3rd Order
    // x* ( x* (A*x + B) + C) + D
    // A :0.1058, B: -0.3942, C: 0.9813, D: 0.006
    // Run above on (x-1) so x is in ln(x+1), plug (x-1 into equation above to
    // save the subtract and get A',B',C',D'):
    // A' = A
    // B' = -3A + B
    // C' = 3a -2B + C
    // D' = -A + B - C + D
    // A':0.1058, B':-0.7116, C':2.0871, D':-1.4753
    ////////////////////////////
    vFloat a = vConstFloatPrgm1;
    vFloat b = vConstFloatPrgm2;
    // XXXXX try variants of the below: B'=.7122, C'=2.0869
    vFloat series_result = x * (x * (x * a + b) + 2.0871) + -1.4753f;

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    vInt exp = exexp(in);
    v_if (exp < 0) {
        exp = setsgn(~exp + 1, 1);
    }
    v_endif;

    vFloat expf = int32_to_float(exp, 0);
    vFloat vConstLn2 = vConstFloatPrgm0;
    vFloat result = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= s2vFloat16a(log_base_scale_factor);
    }

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if (in == 0.0F) { // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    dst_reg[0] = result;
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void calculate_log(uint log_base_scale_factor)
{
    #pragma GCC unroll 8
    for(int d = 0; d < ITERATIONS; d++){
        calculate_log_body<HAS_BASE_SCALING>(log_base_scale_factor);
        dst_reg++;
    }
}

sfpi_inline void calculate_comp_init_flag(bool check, vFloat& flag1, vFloat& flag2, float init)
{
    flag1 = init;
    if (check) {
        flag2 = init;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS>
inline void calculate_comp(uint exponent_size_8)
{
   const vFloat zero = 0.0f;
   const vFloat one = 1.0f;
   for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        vFloat flag1, flag2;

	//a[i] == 0
	if constexpr(COMP_MODE == SfpuType::equal_zero) {
	    v_if (_sfpu_is_fp16_zero_(v, exponent_size_8)) {
	      v = one;
	    } v_else {
	      v = zero;
	    }
	    v_endif;
	  }

	//a[i] != 0
	if constexpr(COMP_MODE == SfpuType::not_equal_zero) {
	    v_if (_sfpu_is_fp16_zero_(v, exponent_size_8)) {
	      v = zero;
	    } v_else {
	      v = one;
	    }
	    v_endif;
        }

	//a[i] < 0
	if constexpr(COMP_MODE == SfpuType::less_than_zero) {
	    v_if (v >= 0.0f) {
	      v = zero;
	    } v_else {
	      v = one;
	    }
	    v_endif;
        }

	//a[i] >= 0
	if constexpr(COMP_MODE == SfpuType::greater_than_equal_zero) {
	    v_if (v >= 0.0f) {
	      v = one;
	    } v_else {
	      v = zero;
	    }
	    v_endif;
        }

	//a[i] > 0
	if constexpr(COMP_MODE == SfpuType::greater_than_zero) {
	    v_if (v > 0.0f) {
	      v = one;
	    } v_else {
	      v = zero;
	    }
	    v_endif;
        }

	//a[i] <= 0
	if constexpr(COMP_MODE == SfpuType::less_than_equal_zero) {
	    v_if (v > 0.0f) {
	      v = zero;
	    } v_else {
	      v = one;
	    }
	    v_endif;
        }

	dst_reg[0] = v;
	dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp(uint param0, uint param1, uint param2)
{
    // All params are in FP16 format
    // param0 = min
    // param1 = max

    //uint format = (param0 >> 16)&0x1;
    s2vFloat16::Format format = s2vFloat16::fp16a;

    // SFPU microcode
    vFloat min = s2vFloat16(param0, format);
    vFloat max = s2vFloat16(param1, format);
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        v_if (val < min) {
            val = s2vFloat16(param0, format);
        } v_elseif (val >= max) {
            val = s2vFloat16(param1, format);
        }
        v_endif;

        dst_reg[0] = val + s2vFloat16b(param2); // 12 bits

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_abs()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}


template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_exp2()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        // log(2) = 0.6931471805;
        v = v * 0.6931471805f;
	    // exp = e^(v)
	    vFloat exp = calculate_exponential_body_improved<APPROXIMATION_MODE, true>(v);
	    dst_reg[0] = exp;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sign()
{
    // All params are in FP16 format
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
	vFloat result = vConst1;
        v_if (v < 0.0f) {
           result = vConstNeg1;
        } v_elseif(v > 0.0f) {
	  result = vConst1;
	} v_else {
	  result = vConst0;
        }
        v_endif;

	dst_reg[0] = result;
        dst_reg++;
    }
}
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_max()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat a = dst_reg[0];
        vFloat b = dst_reg[32];
        v_if(a < b) {
            dst_reg[0] = b;
        }
        v_endif;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_min()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat a = dst_reg[0];
        vFloat b = dst_reg[32];
        v_if(a > b) {
            dst_reg[0] = b;
        }
        v_endif;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_expm1()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        v = calculate_exponential_body_improved<APPROXIMATION_MODE, true>(v);
        dst_reg[0] = v - 1.0f;
        dst_reg++;
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

template <bool APPROXIMATION_MODE, int ITERATIONS>
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

template <bool APPROXIMATION_MODE, int ITERATIONS>
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


#define PI_2 (1.570796326794)
template <bool APPROXIMATION_MODE, int ITERATIONS>
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

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void cast_fp32_to_fp16a()
{
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        //vFloat val = dst_reg[0];
        //dst_reg[0] = float_to_fp16a(val, 0);
        TTI_SFPLOAD(0, 0, 3, 0);
        TTI_SFP_STOCH_RND(0,0,0,0,0,8);
        TTI_SFPSTORE(0,1,3,0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_add1()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        dst_reg[0] = 1.0f + val;
        dst_reg++;
    }
}

inline
vFloat sigmoid_piecewise_linear_positive(vFloat val) {
        vFloat result = 0.0f;
	v_if ( val >= +5.0f)  {
	  result = 1.0f;
	} v_elseif ( val > 1.0f && val < 5.0f ) {
	  result = POLYVAL5(0.00144462f, -0.01055479f, -0.01203685f,  0.24300185f,  0.50437757f,val);
	} v_else {
	  result = 0.229f*val + 0.5f; // linear appx as y = 0.229x + 0.5
	}
	v_endif;
	return result;
}

//sigmoid is anti-symmetric and offset by 1
//sigmoid[-x] = 1 - sigmoid[x]
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sigmoid()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat result = 0.0f;

        v_if ( val < 0.0f ) {
  	   val = -val;
        }
        v_endif;

	result = sigmoid_piecewise_linear_positive(val);

	val = dst_reg[0];
        v_if ( val < 0.0f ) {
            result = 1.0f - result;
        }
        v_endif;

        dst_reg[0] = result;
        dst_reg++;
    }

    return;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_heaviside(uint value)
{
    // SFPU microcode
    Converter c_value;
    c_value.u = value;
    vFloat s = c_value.f;

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        v_if (v < 0.0f) {
            v = 0.0f;
        }v_elseif (v > 0.0f) {
            v = 1.0f;
        }v_else {
            v = s;
        }
        v_endif;

       dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_silu()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        v_if ( val < 0.0f ) {
            val = -val;
        }
        v_endif;

	    vFloat result = sigmoid_piecewise_linear_positive(val);

	    val = dst_reg[0];
        v_if ( val < 0.0f ) {
            result = 1.0f - result;
        }
        v_endif;
        result = val * result;
        dst_reg[0] = result;
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
