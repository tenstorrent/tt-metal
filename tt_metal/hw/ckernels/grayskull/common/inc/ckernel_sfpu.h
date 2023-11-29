// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_cdf.h"
#include "ckernel_sfpu_gelu.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_recip.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

sfpi_inline vInt sfpu_is_fp16_zero(const vFloat& v, uint exponent_size_8)
{
    if (exponent_size_8) {
        // fp16b
        return v == 0.0F;
    } else {
        // fp16a
        // if math data format is fp16, SFPU will convert 5 bit exp to 8 bit exp
        // in grayskull, this unconditionally adds bias value to exp (even for zero)
        vInt tmp = 0x3800; // loads {0, 8'd112, 10'b0}
        tmp += reinterpret<vInt>(v);

        return tmp == 0;
    }
}

inline void init_dropout_seed(uint16_t p2)
{
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);

    uint16_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint16_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

    uint16_t per_tensix_input_seed = p2 ^ (my_x << my_y);

    vInt result = l_reg[LRegs::LReg3];

    vInt tmp = vConstTileId << 13;
    vInt ptis = reinterpret<vInt>(vFloat(per_tensix_input_seed));
    result = ~(tmp & ptis) & (tmp | ptis);

    l_reg[LRegs::LReg3] = result;
}

template <bool APPROXIMATION_MODE>
inline void sfpu_init(SfpuType operation, uint param0 = 0)
{
    uint imm0;
    uint imm1;
    uint imm2;
    switch (operation) {
    case SfpuType::tanh:
    case SfpuType::tanh_derivative:
        imm0 = 0x1DFF; //0.90625*x
        imm1 = 0x481A; //0.09375*x + 0.8125
        imm2 = 0xFF00; //1
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
        TTI_SFPLOADI(2, 2, imm2);
        break;
    case SfpuType::sigmoid_appx:
        imm0 = 0x3DFF;
        imm1 = 0x21D8;
        imm2 = 0xFF10;
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
        TTI_SFPLOADI(2, 2, imm2);
        break;
    case SfpuType::sigmoid:
        break;
    case SfpuType::expm1:
    case SfpuType::exp2:
        if constexpr(APPROXIMATION_MODE) {
            TTI_SFPLOADI(p_sfpu::LREG2, 0, p_exp::ADJ_EXP);
        }
        break;
    case SfpuType::dropout:
        init_dropout_seed(param0);
        break;
    default:
        // Should result in compile time error??
        break;
    }
}

/*
template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN>
void calculate_cube(uint16_t exp_base_scale_factor = 0)
{
    for (int d = 0; d < ITERATIONS; d++)
    {

        TTI_SFPLOAD(p_sfpu::LREG3, 0, 0); // load from dest
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_NOP; TTI_NOP;
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);
        TTI_NOP; TTI_NOP;
        TTI_SFPSTORE(p_sfpu::LREG2, 0, 0); // Store from lreg[1] into dest registers
        TTI_INCRWC(0, 4, 0, 0);
    }
}
*/



template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_expm1()
{
   // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        vFloat out = calculate_exponential_body_improved<APPROXIMATION_MODE, true>(v);
        dst_reg[0] = out - 1.0f;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat sfpu_asine_maclaurin_series(vFloat val)
{
    // input for [-1:1]
    // Mclauren series
    // arcsin(x) = x + [(1/2) *x^3/3] + [(1 * 3) / (2 * 4) * x^5 / 5] + [(1 * 3 * 5) / (2 * 4 * 6) * x^7 / 7 ] + ...
    // arcsin(x) ≈ x + (1/6) * x^3 + (3/40) * x^5 + (5/112) * x^7 + (35/1152) * x^9 + (63/2816) * x^11

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




template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_acos()
{
    // SFPU microcode
    // acos = (pi/2 - asin)
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        v = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v);
        v = (1.570796326794) - v;
        dst_reg[0] = v;
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
        dst_reg[0] =  sfpu_reciprocal<true>(sfpi::abs(val));
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


template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_negative()
{

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        dst_reg[0] = -val;
        dst_reg++;
    }
}


template <bool APPROXIMATION_MODE, int ITERATIONS, int RECIPROCAL_ITERATIONS>
inline void calculate_rsqrt()
{

    for (int d = 0; d < ITERATIONS; d++)
    {

        vFloat in = dst_reg[0];
        vFloat result = sfpu_reciprocal<false>(in);
        v_if(dst_reg[0] < 1.0f){
            result = 1.0f;
        }v_endif;

        for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
        {
            // y = y * (1.5 - 0.5 * x * y * y) Newton's method iteration.
            result = result * (1.5F - 0.5F  * dst_reg[0] * result * result);
        }

        v_if(dst_reg[0] == 0.0f){
            dst_reg[0] = std::numeric_limits<float>::infinity();
        }v_else{
            dst_reg[0] = result;
        }v_endif;

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

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_sigmoid_appx()
{
    // SFPU microcode
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        val = lut(val, l0, l1, l2);

        dst_reg[0] = val + 0.5f;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_tanh()
{
    // SFPU microcode
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

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

template <bool APPROXIMATION_MODE>
inline void calculate_hardtanh(uint param0, uint param1, uint param2, int ITERATIONS)
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

        val = val * val;
        val = vConst1 - val;
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
    vUInt mask = reinterpret<vUInt>(vFloat(s2vFloat16b(0xa94b)));

    #pragma GCC unroll 0
    for (int d=0; d<4; d++) {
        ////////////////////////
        // Scale samples
        ///////////////////////
        dst_reg[0] = dst_reg[0] * s2vFloat16b(scale);

        ////////////////////////
        // Drop samples
        ///////////////////////
        vUInt tmp = rand >> 3;
        v_if (tmp < vUInt(prob)) {
            dst_reg[0] = vConst0;
        }
        v_endif;

        ////////////////////////
        // 16-bit PRNG update
        ///////////////////////
        tmp = rand << 1;

        // Mask = 0x593CA -> 29e4d
        // Mask = 0xd295 -> a94b
        // PRNG SHL by one
        v_if (tmp < 0) {
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
    #pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++)
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
    #pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat in = dst_reg[0];
        vFloat result = in * in;

        dst_reg[0] = result;

        dst_reg++;
    }
}

template <bool HAS_BASE_SCALING>
sfpi_inline void calculate_log_body(const int log_base_scale_factor)
{
    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    vFloat in = dst_reg[0];
    vFloat x = setexp(in, 127);    // set exp to exp bias (put in range of 1-2)

    ////////////////////////////
    // Calculate Cheby Approximation using Horner Form Multiplication: 3rd Order
    // x* ( x* (A*x + B) + C) + D
    // A :0.1058, B: -0.3942, C: 0.9813, D: 0.006
    // Run above on (x-1) so x is in ln(x+1), plug (x-1 into equation above to
    // save the subtract and get A',B',C',D':
    // A' = A
    // B' = -3A + B
    // C' = 3a -2B + C
    // D' = -A + B - C + D
    // A':0.1058, B':-0.7116, C':2.0871, D':-1.4753
    ////////////////////////////
    vFloat a = s2vFloat16a(0.1058F);
    vFloat series_result = x * (x * (x * a + s2vFloat16a(-0.7122f)) + s2vFloat16a(2.0869)) + s2vFloat16a(-1.4753f);

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    // Extract exponent and calculate abs value.  Save sign into partial reg
    vInt exp = 0;
    v_if (in != 0.0F) {
        exp = exexp(in);
        v_if (exp < 0) {
            exp = sfpi::abs(exp);
            in = setsgn(in, 1);
        }
        v_endif;
    }
    v_endif;

    // Calculate exponent of the exponent value. Done by using LZ
    // Get leading zero.  If not zero, we do 19 + ~LZ to get exponent value (mathematically == 19 - LZ - 1)
    vInt new_exp = 0;
    v_if (exp != 0) {
        new_exp = lz(exp);
        new_exp = ~new_exp;
        new_exp += 19;
        v_if (new_exp >= 0) {
            new_exp += 127;
        }
        v_endif;
    }
    v_endif;

    vFloat result = setexp(in, new_exp);
    vInt shift = lz(exp) + 1;
    result = setman(result, shft(reinterpret<vUInt>(exp), shift));
    result = result * vConst0p6929 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= s2vFloat16a(log_base_scale_factor);
    }

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if (dst_reg[0] == 0.0F) { // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    dst_reg[0] = result;

    dst_reg++;
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void calculate_log(uint log_base_scale_factor)
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_log_body<HAS_BASE_SCALING>(log_base_scale_factor);
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
	    v_if (sfpu_is_fp16_zero(v, exponent_size_8)) {
	      v = one;
	    } v_else {
	      v = zero;
	    }
	    v_endif;
	  }

	//a[i] != 0
	if constexpr(COMP_MODE == SfpuType::not_equal_zero) {
	    v_if (sfpu_is_fp16_zero(v, exponent_size_8)) {
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


template <bool APPROXIMATION_MODE>
inline void calculate_clamp(uint param0, uint param1, uint param2, int ITERATIONS)
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
    constexpr bool zero_negative = true;
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        // y = exp(x * log(2))
        // log(2) = 0.6931471805;
        v = v * 0.6931471805f;
        vFloat exp = calculate_exponential_body_improved<APPROXIMATION_MODE, zero_negative>(v);
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
        vFloat b = dst_reg[16];
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
        vFloat b = dst_reg[16];
        v_if(a > b) {
            dst_reg[0] = b;
        }
        v_endif;

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

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_mask()
{
    bool exponent_size_8 = true;
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat mask = dst_reg[16];
        v_if(sfpu_is_fp16_zero(mask, exponent_size_8)) {
            dst_reg[0] = 0;
        }
        v_endif;
        dst_reg++;
    }
}

template <SfpuType operation, bool APPROXIMATION_MODE, int SfpuType_PARAM = 0, int ITERATIONS = 4>
inline void calculate_sfpu(uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0)
{
    if constexpr (operation == SfpuType::exp_with_base) {
	    constexpr bool zero_negative = true;
        calculate_exponential<APPROXIMATION_MODE, zero_negative, true, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::tanh) {
        calculate_tanh<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::signbit) {
        calculate_signbit<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::hardtanh) {
        calculate_hardtanh<APPROXIMATION_MODE, ITERATIONS>(param0, param1, param2);
    }
    else if constexpr (operation == SfpuType::rsqrt) {
	//param0 = true -> approximate fast mode
	//         false -> high precision mode
    // The algorithm uses Newton's method based on no.of iteration better approximation can be calculated
	if ( param0 ) {
	    calculate_rsqrt<true, ITERATIONS, 10>();
	} else {
	    calculate_rsqrt<false, ITERATIONS, 25>();
	}
    }
    else if constexpr (operation == SfpuType::sigmoid) {
        calculate_sigmoid<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::sigmoid_appx) {
        calculate_sigmoid_appx<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::tanh_derivative) {
        calculate_tanh_derivative<APPROXIMATION_MODE, SfpuType_PARAM, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::dropout) {
        calculate_dropout<APPROXIMATION_MODE, ITERATIONS>(param0, param1);
    }
    else if constexpr (operation == SfpuType::power) {
	calculate_power_iterative<APPROXIMATION_MODE, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::square) {
        calculate_square<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::log) {
        calculate_log<APPROXIMATION_MODE, false, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::log_with_base) {
        calculate_log<APPROXIMATION_MODE, true, ITERATIONS>(param0);
    }
    else if constexpr ((operation == SfpuType::equal_zero) ||
                       (operation == SfpuType::not_equal_zero) ||
                       (operation == SfpuType::less_than_zero) ||
                       (operation == SfpuType::greater_than_equal_zero) ||
                       (operation == SfpuType::less_than_equal_zero) ||
                       (operation == SfpuType::greater_than_zero)) {
        calculate_comp<APPROXIMATION_MODE, operation, ITERATIONS>(8); //BFLOAT16 - exp
    }
    else if constexpr (operation == SfpuType::clamp) {
        calculate_clamp<APPROXIMATION_MODE, ITERATIONS>(param0, param1, param2);
    }
    else if constexpr (operation == SfpuType::abs) {
        calculate_abs<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::sign) {
        calculate_sign<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::max) {
        calculate_max<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::min) {
        calculate_min<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::exp2) {
        calculate_exp2<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::heaviside) {
        calculate_heaviside<APPROXIMATION_MODE, ITERATIONS>(param0);
    }
    else if constexpr (operation == SfpuType::expm1) {
        calculate_expm1<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::asin) {
        calculate_asin<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::acos) {
        calculate_acos<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::atan) {
        calculate_atan<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::silu) {
        calculate_silu<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::mask) {
        calculate_mask<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::negative) {
        calculate_negative<APPROXIMATION_MODE, ITERATIONS>();
    }

    //erf, erfc are dispatched directly.
}

} // namespace sfpu
} // namespace ckernel
