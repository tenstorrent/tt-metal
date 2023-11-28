/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"
#include "llk_defs.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{
    
sfpi_inline vInt _sfpu_is_fp16_zero_(const vFloat& v, uint exponent_size_8)
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

sfpi_inline vFloat _sfpu_exp_(vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    vInt exp = exexp(val);
    v_if (exp >= 0) {
        val = setexp(val, 126);
    }
    v_endif;

    // Run series in Horner form
    vFloat tmp = val * vConst0p8369 + 0.8634F;
    val = val * tmp + vConst1;

    v_if (exp >= 0) {
        val = val * val;
        #pragma GCC unroll 0
        for (int s_iter = 0; s_iter < 7; s_iter++) {
            exp = exp - 1;
            // Narrow predication on each loop
            v_and(exp >= 0);
            val = val * val;
        }
    }
    v_endif;

    return val;
}

template <bool save_reg, int max_iter = 3>
sfpi_inline vFloat _sfpu_reciprocal_(const vFloat in)
{
    vInt orig_exp;

    if constexpr (max_iter == 1) {
        // If we are only doing one iteration of the MAD loop, then we only need to use one LREG for the MAD instructions because we have our "first guess" in a hard-coded register
        // This allows us to avoid having to load back in the original value later on
        orig_exp = exexp(in);
    }

    // Force sign to 1 (make number negative)
    vFloat val = setsgn(in, 1);

    val = setexp(val, 126); // Set exponent to 126 to make the number in 0.5-1
    // Use 1.44 as first guess at x, ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
    vFloat two;
    if (!save_reg) {
        two = 2.0f;
    }
    vFloat result = vConst1p4424 * (val * vConst1p4424 + (save_reg ? 2.0f : two));

    for (int s_iter = 0; s_iter < (max_iter-1); s_iter++) {
        result = result * (val * result + (save_reg ? 2.0f : two));
    }

    vInt new_exp = exexp(result);
    if constexpr (max_iter != 1) {
        orig_exp = exexp(dst_reg[0]);
    }

    // "Subtract" exponents, and re-bias.  
    // Execute: -1 - exp, then exp += 127
    new_exp -= orig_exp;
    new_exp += 126;

    v_if (new_exp < 0) {
        // If rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        result = 0.0F;
        new_exp = 0;
    }
    v_endif;

    // Set newly denormalized exponent to result exponent field
    return setexp(result, new_exp);
}

inline void _init_dropout_seed_(uint16_t p2)
{
    FWLOG1("calculate_dropout() -- input seed:%x", p2);
    
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);

    uint16_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint16_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

    uint16_t per_tensix_input_seed = p2 ^ (my_x << my_y);

    FWLOG1("calculate_dropout() -- calculated seed:%x", per_tensix_input_seed);
    
    vInt result = l_reg[LRegs::LReg3];

    vInt tmp = vConstTileId << 13;
    vInt ptis = reinterpret<vInt>(vFloat(per_tensix_input_seed));
    result = ~(tmp & ptis) & (tmp | ptis);

    l_reg[LRegs::LReg3] = result;
}

template <bool APPROXIMATION_MODE>
inline void _init_exponential_()
{
    if constexpr(APPROXIMATION_MODE) {
        TTI_SFPLOADI(p_sfpu::LREG0, 0, p_exp::C23_73);
        TTI_SFPLOADI(p_sfpu::LREG2, 0, p_exp::ADJ_EXP);
    }
} 

template <bool APPROXIMATION_MODE>
inline void _init_sqrt_() 
{
    if (APPROXIMATION_MODE) {
        TTI_SFPLOADI(2, 0, 127 << 7);
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_tanh_() 
{
    uint imm0;
    uint imm1;
    uint imm2;
    imm0 = 0x1DFF; //0.90625*x
    imm1 = 0x481A; //0.09375*x + 0.8125
    imm2 = 0xFF00; //1
    TTI_SFPLOADI(0, 2, imm0);
    TTI_SFPLOADI(1, 2, imm1);
    TTI_SFPLOADI(2, 2, imm2);
}

template <bool APPROXIMATION_MODE>
inline void _init_sigmoid_() 
{
    uint imm0;
    uint imm1;
    uint imm2;
    imm0 = 0x3DFF;
    imm1 = 0x21D8;
    imm2 = 0xFF10;
    TTI_SFPLOADI(0, 2, imm0);
    TTI_SFPLOADI(1, 2, imm1);
    TTI_SFPLOADI(2, 2, imm2);
}

template <bool APPROXIMATION_MODE>
inline void _init_gelu_() 
{
    imm0 = 0x18FF;
    imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
    imm2 = 0xFF00;
    TTI_SFPLOADI(0, 2, imm0);
    TTI_SFPLOADI(1, 2, imm1);
    TTI_SFPLOADI(2, 2, imm2);
}

inline void _init_dropout_(const uint seed) 
{
    _init_dropout_seed_(seed);
}

template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE>
sfpi_inline vFloat _calculate_exponential_body_(vFloat in)
{
    vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        // * by 1/ln2 and add convert to 7.3 FxP format
        vFloat val = in * vConst1p4424 + p_exp::C23_73;

        // Remove Exponent of 7 and bias the Mantissa to 127.
        // LREG2 already holds 2's complement value so we simply do REG2 + REG3 
        vInt val_short = p_exp::ADJ_EXP + reinterpret<vInt>(val);

        // SHL to move integer bits to exponent
        val_short <<= 10 - p_exp::FRAC_BITS;
        out = reinterpret<vFloat>(val_short);

        // Needed for fused kernels such as math_row_softmax_tables which call calculate_exponential()
        // without using Relu in Packer to clamp -ve Infinity to 0.
        if constexpr (ZERO_NEGATIVE)
        {
            v_if (val_short < 0) {
                out = vConst0;
            }
            v_endif;
        }
    }
    else
    {
        // Force sign to 0 (make number positive)
        vFloat exp = _sfpu_exp_(setsgn(in, 0));

        // Load input value, to determine whether reciprocal needs to be run
        vFloat val = dst_reg[0];

        // store tentatively e^x
        // reciprocal function relies on reloading input
        dst_reg[0] = exp;

        v_if (val < 0) {
            dst_reg[0] = _sfpu_reciprocal_<true>(exp);
        }
        v_endif;
    }
    return out;
}

template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN, int ITERATIONS>
inline void _calculate_exponential_(int16_t exp_base_scale_factor = 0)
{
    vFloat c23_73;
    vInt adj_exp;

    if constexpr (APPROXIMATION_MODE)
    {
        c23_73 = l_reg[LRegs::LReg0];
        adj_exp = l_reg[LRegs::LReg2];
    }

    #pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        if constexpr(SCALE_EN){
            val = val * s2vFloat16a(exp_base_scale_factor);
            dst_reg[0] = val;
        }
        if constexpr (APPROXIMATION_MODE)
        {
            v_if (val>=89){
                vFloat val_inf = std::numeric_limits<float>::infinity();
                dst_reg[0] = val_inf;
            } v_elseif(val<-42){
                    dst_reg[0] = 0.0f;
            } v_else {
                // * by 1/ln2 and add convert to 7.3 FxP format
                val = val * vConst1p4424 + c23_73;

                // Remove Exponent of 7 and bias the Mantissa to 127.
                // LREG2 already holds 2's complement value so we simply do REG2 + REG3 
                vInt val_short = adj_exp + reinterpret<vInt>(val);

                // SHL to move integer bits to exponent
                val_short <<= 10 - p_exp::FRAC_BITS;
                dst_reg[0] = reinterpret<vFloat>(val_short);
            }
            v_endif;
        }
        else
        {
            // Force sign to 0 (make number positive)
            val = _sfpu_exp_(setsgn(val, 0));

            vFloat orig = dst_reg[0];

            // Loaded by reciprocal
            dst_reg[0] = val;
            v_if (orig < 0) {
                dst_reg[0] = _sfpu_reciprocal_<false>(val);
            }
            v_endif;
        }

        dst_reg++;
    }

    if constexpr (APPROXIMATION_MODE)
    {
        l_reg[LRegs::LReg0] = c23_73;
        l_reg[LRegs::LReg2] = adj_exp;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _calculate_gelu_core_(vFloat in)
{
    constexpr uint imm0 = 0x18FF;
    constexpr uint imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
    constexpr uint imm2 = 0xFF00;

    // SFPU microcode: 
    // result = (APPROX_MODE == 1) 
    //   ? (1 + erf(x/sqrt(2)))
    //   : (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) )
    vFloat result;
    if constexpr (APPROXIMATION_MODE) {
        result = in;
    } else {
        // f = (0.044715*x^3 + x)
        result = in * in * in;
        result = result * 0.044715f + in;

        result *= 0.79788f;
    }

    result = lut(result, imm0, imm1, imm2);

    result = result * 0.5f + 0.5f;

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_gelu_()
{
    constexpr uint imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
    constexpr uint imm2 = 0xFF00;
    vUInt l0 = l_reg[LRegs::LReg0];

    // SFPU microcode
    #pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vUInt l1;
        vUInt l2;
        vFloat result;

        if constexpr (APPROXIMATION_MODE)
        {
            l1 = imm1;
            l2 = imm2;
            result = val;
        } else {
            // f = (0.044715*x^3 + x)
            result = (val * val * val) * 0.044715f + val;

            // result = result * sqrt(2/pi)
            result *= 0.7969f;

            // Reload l1, l2 for lut
            l1 = imm1;
            l2 = imm2;
        }

        result = lut(result, l0, l1, l2);

        val = dst_reg[0];

        result = val * result + val;
        result *= 0.5f;

        dst_reg[0] = result;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sigmoid_()
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
inline void _calculate_tanh_()
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

template <bool APPROXIMATION_MODE>
inline void _calculate_hardtanh_(uint param0, uint param1, uint param2, int ITERATIONS)
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
inline void _calculate_tanh_derivative_()
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
inline void _calculate_gelu_derivative_()
{
    // SFPU microcode: 
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat result = val * val * vConstNeg0p5;
        
        // exp = e^(val) * 1/sqrt(2*pi)
        if constexpr(APPROXIMATION_MODE) {
            vFloat exp = _calculate_exponential_body_<APPROXIMATION_MODE, APPROXIMATION_MODE>(result);
            exp *= 0.39844F;
            dst_reg[0] = exp * val;
        } else {
            dst_reg[0] = result;
            _calculate_exponential_body_<APPROXIMATION_MODE, APPROXIMATION_MODE>(result);
            vFloat exp = dst_reg[0];
            exp *= 0.39844F;
            dst_reg[0] = exp * val;
        }
        result = _calculate_gelu_core_<APPROXIMATION_MODE>(val);

        dst_reg[0] = dst_reg[0] + result;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_reciprocal_()
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat in = dst_reg[0];
        vFloat out = _sfpu_reciprocal_<false, APPROXIMATION_MODE ? 2 : 3>(in);

        // Reload to reduce register pressure
        v_if (dst_reg[0] < 0.0F) {
            // Invert sign on calculated value if CC=1 (number is negative)
            out = -out;
        }
        v_endif;

        dst_reg[0] = out;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, int RECIPROCAL_ITERATIONS=2>
inline void _calculate_sqrt_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];

        if constexpr (APPROXIMATION_MODE)
        {
            vUInt magic = l_reg[LRegs::LReg2];

            //sqrt initial approximation
            // adjust bias
            vUInt val_s = magic + reinterpret<vUInt>(val);

            // approximation of square root
            val_s >>= 1;
            dst_reg[0] = reinterpret<vFloat>(val_s);

            l_reg[LRegs::LReg2] = magic;
        }
        else
        {
            // Recip root method
            //// Init approx
            //u.i = SQRT_MAGIC_F - (u.i >> 1);
            vUInt magic = reinterpret<vUInt>(vFloat(s2vFloat16b(0x5f37)));
            vFloat approx = reinterpret<vFloat>(magic - (reinterpret<vUInt>(val) >> 1));

            // Re-load to save a MOV
            val = dst_reg[0];

            //Reciproot iterations
            for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
            {
                //x*r*(1.5f - xhalf*r*r);
                approx = (approx * approx * val * vConstNeg0p5 + vConst1 + 0.5F) * approx;
            }

            dst_reg[0] = approx * val;
        }

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_dropout_(uint prob, uint scale)
{
    // SFPU microcode

    FWLOG1("calculate_dropout() -- prob:%x", prob);
    FWLOG1("calculate_dropout() -- scale:%x", scale);

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

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_lrelu_(uint slope)
{
    // SFPU microcode
    vFloat s = s2vFloat16b(slope);

    #pragma GCC unroll 0
    for (int d=0; d<4; d++) {
        vFloat v = dst_reg[0];

        v_if (v < 0.0f) {
            v *= s;
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_power_(uint exponent)
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat in = dst_reg[0];
        vFloat result = in * in;
        for (uint i = 2; i < exponent; i++) {
            result *= in;
        }

        dst_reg[0] = result;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_square_()
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
sfpi_inline void _calculate_log_body_(const int log_base_scale_factor)
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
inline void _calculate_log_(uint log_base_scale_factor)
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_log_body_<HAS_BASE_SCALING>(log_base_scale_factor);
    }
}

sfpi_inline void _calculate_comp_init_flag_(bool check, vFloat& flag1, vFloat& flag2, float init)
{
    flag1 = init;
    if (check) {
        flag2 = init;
    }
}

template <bool APPROXIMATION_MODE, bool invert_output, bool check_zero, bool second_check, bool is_less_than_equal_zero, int ITERATIONS>
inline void _calculate_comp_(uint exponent_size_8)
{
    // output_0 and output_1 hold the outputs use use when a zero or negative check is true/false.
    // False = 0.0 = kCONST_0 (5/8-bit exponent format)
    // True  = 1.0 = kCONST_1_FP16B (8-bit exponent format)
    // SFPU uses 8-bit exponent in operations so loading these constants in 8-bit exponent format.
    // Although a command flag can tell SFPU to re-bias a 5-bit exponent to 8-bit, we are loading 8-bit 
    // exponent and telling SFPU to not add any bias to these constants.
    constexpr float output_0 = invert_output ? 0.0f : 1.0f;
    constexpr float output_1 = invert_output ? 1.0f : 0.0f;

    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        vFloat flag1, flag2;
        if constexpr(check_zero)
        {
            v_if (_sfpu_is_fp16_zero_(v, exponent_size_8)) {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_0);
            } v_else {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_1);
            }
            v_endif;
        }
        else
        {
            v_if (v < 0.0F) {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_0);
            } v_else {
                _calculate_comp_init_flag_(second_check, flag1, flag2, output_1);
            }
            v_endif;
        }

        vFloat result;
        if constexpr (second_check)
        {
            // SfpuType::less_than_equal_zero
            // flag1 = 0x3F80(1.0) if DST < 0 else 0
            // flag2 = 0x3F80(1.0) if DST == 0 else 0
            // Do a bitwise Or (flag1 | flag2) to get <= condition.
            // flag1 < 0 OR flag2 == 0 => DST is Less than or Equal to zero.
            // Result will be either 0x0000(0.0) or 0x3F80(1.0)
            if constexpr (is_less_than_equal_zero) {
                result = reinterpret<vFloat>(reinterpret<vUInt>(flag1) | reinterpret<vUInt>(flag2));
            }
            else
            {
                // SfpuType::greater_than_zero
                // flag1 = 0x3F80(1.0) if DST >= 0 else 0
                // flag2 = 0x3F80(1.0) if DST != 0 else 0
                // Do a bitwise And (flag1 & flag2) to get > condition.
                // flag2 >= 0 AND flag1 != 0 => DST is Greater than zero 
                // Result will be either 0x0000(0.0) or 0x3F80(1.0)
                result = reinterpret<vFloat>(reinterpret<vUInt>(flag1) & reinterpret<vUInt>(flag2));
            }
        } else {
            result = flag1;
        }

        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _calculate_clamp_(uint param0, uint param1, uint param2, int ITERATIONS)
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
inline void _calculate_abs_()
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
inline void _calculate_sign_(uint exponent_size_8)
{
    // All params are in FP16 format
    // uint format = 1;
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];
        dst_reg[0] = vConst1;
        v_if (v < 0.0F) {
            dst_reg[0] = vConstNeg1;
        }
        v_endif;

        //param0 == 0 is Bfp8 format. It does not require bias removal.
        //param0 != 0 is Float16 format and exp bias needs to be removed for zero check.
        v_if (_sfpu_is_fp16_zero_(v, exponent_size_8)) {
            dst_reg[0] = vConst0;
        }
        v_endif;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_max_()
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
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _sfpu_sine_maclaurin_series_(vFloat val)
{
    // Good for [-pi:pi]
    // Mclauren series = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    vFloat tmp = val;
    // x
    vFloat output = tmp;
    // x^3/3!
    tmp = tmp*val*val;
    output += -0.166666666*tmp;
    // x^5/5!
    tmp = tmp*val*val;
    output +=  0.0083333333*tmp;
    // x^7/7!
    tmp = tmp*val*val;
    output += -0.0001984126*tmp;
    if constexpr (not APPROXIMATION_MODE) {
        // x^9/9!
        tmp = tmp*val*val;
        output +=  0.0000027557*tmp; 
        // x^11/11!
        tmp = tmp*val*val;
        output += -0.00000002505*tmp;
    }

    // Write out output
    return output;
}
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _sfpu_cosine_maclaurin_series_(vFloat val)
{
    // Good for [-pi:pi]
    // Mclauren series = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12!
    // 1
    vFloat output = 1.0f;
    // x^2/2!
    vFloat tmp = val*val;
    output += -0.5*tmp;
    // x^4/4!
    tmp = tmp*val*val;
    output +=  0.0416666666*tmp;
    // x^6/6!
    tmp = tmp*val*val;
    output += -0.0013888888*tmp;
    if constexpr (not APPROXIMATION_MODE) {
        // x^8/8!
        tmp = tmp*val*val;
        output +=  0.0000248015*tmp;
        // x^10/10!
        tmp = tmp*val*val;
        output += -0.0000002755*tmp;
    }

    // Write out output
    return output;
}
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sine_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];

        // Assume v is bound [0:2pi]
        // phase shift [0:2pi] to [-pi:pi] and multiply result by -1 
        v = v - 3.14159264f;
        v = _sfpu_sine_maclaurin_series_<APPROXIMATION_MODE>(v);

        // Use symmetrical properties of trig
        v *= -1;

        // Write Output
        dst_reg[0] = v;
        dst_reg++;
    }
}
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_cosine_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat v = dst_reg[0];

        // Assume v is bound [0:2pi]
        // phase shift [0:2pi] to [-pi:pi] and multiply result by -1 
        v = v - 3.14159264f;
        v = _sfpu_cosine_maclaurin_series_<APPROXIMATION_MODE>(v);

        // Use symmetrical properties of trig
        v *= -1;

        // Write Output
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_(uint uint_threshold)
{
    vFloat threshold = s2vFloat16(uint_threshold, s2vFloat16::fp16a);
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat a = dst_reg[0];
        v_if(a > threshold) { 
            a = threshold; 
        }
        v_endif;
        v_if(a < 0.0f) { 
            a = 0.0f; 
        }
        v_endif;
        dst_reg[0] = a; 
        dst_reg++;
    }
}
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_(uint uint_threshold)
{
    vFloat threshold = s2vFloat16(uint_threshold, s2vFloat16::fp16a);
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat a = dst_reg[0];
        v_if(a < threshold) { 
            a = 0.0f;
        }
        v_endif;
        dst_reg[0] = a; 
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
