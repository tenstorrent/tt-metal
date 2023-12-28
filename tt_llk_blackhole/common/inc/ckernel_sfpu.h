// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel.h"
#include <limits>

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

inline void set_dst_write_addr(uint32_t addr) {
    uint dst_index = addr + get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
}

inline void _sfpu_load_imm32_(const uint dest, const uint val)
{
        TT_SFPLOADI(dest, 10, (val & 0xFFFF));  // insmod == 10 will write the lower bits, and not affect the upper bits;
        TT_SFPLOADI(dest, 8, (val>>16) & 0xFFFF);  // insmod == 8 will write the upper bits, and not affect the lower bits;
}

inline void _sfpu_load_imm16_(const uint dest, const uint val)
{
        TT_SFPLOADI(dest, 2, val);  // insmod == 2 will write imm16 value treated as unsigned integer, right justified and padded with zeroes on the MSBs
}

inline void _sfpu_load_config32_(const uint dest, const uint upper16, const uint lower16)
{
        // registers 11 through 14 are programmable "constants" which are shared across all 4 rows
        // They are updated only through the CONFIG path, which uses LREG[0] first and then copies it to the desired register location
        TTI_SFPLOADI(0, 10, lower16);  // insmod == A will write the lower bits, and not affect the upper bits;
        TTI_SFPLOADI(0, 8, upper16);  // insmod == 8 will write the upper bits, and not affect the lower bits;
        TTI_SFPCONFIG(0, dest, 0);
}

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
    vFloat tmp = val * vConst0p8373 + s2vFloat16b(0.863281);
    val = val * tmp + vConst1;

    v_if (exp >= 0) {
        val = val * val;
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

template <int max_iter = 3>
sfpi_inline vFloat _sfpu_reciprocal_(const vFloat in)
{
    // Force sign to 1 (make number negative)
    vFloat val = setsgn(in, 1);

    val = setexp(val, 126); // Set exponent to 126 to make the number in 0.5-1
    // Use 1.44 as first guess at x, ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
    vFloat vConstLn2Recip = vConstFloatPrgm0;
    vFloat two = vConstFloatPrgm1;
    vFloat result = vConstLn2Recip * (val * vConstLn2Recip + two);

    for (int s_iter = 0; s_iter < (max_iter-1); s_iter++) {
        result = result * (val * result + two);
    }

    vInt orig_exp = exexp(in);
    vInt new_exp = exexp(result);

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

inline void _init_dropout_seed_(uint16_t p2){
    FWLOG1("calculate_dropout() -- input seed:%x", p2);

    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);

    uint16_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint16_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

    uint16_t per_tensix_input_seed = p2 ^ (my_x << my_y);

    FWLOG1("calculate_dropout() -- calculated seed:%x", per_tensix_input_seed);

    vInt result = l_reg[LRegs::LReg3];

    vInt tmp = vConstTileId << 10;
    vInt ptis = per_tensix_input_seed;
    result = ~(tmp & ptis) & (tmp | ptis);

    l_reg[LRegs::LReg3] = result;
}

template <bool APPROXIMATION_MODE>
inline void _init_exponential_()
{
    if (APPROXIMATION_MODE) {
        vConstFloatPrgm0 = 1.442695f; // ln2_recip
        vConstFloatPrgm1 = s2vFloat16b(p_exp::C23_73);
        vConstFloatPrgm2 = s2vFloat16b(p_exp::ADJ_EXP);
    } else {
        vConstFloatPrgm0 = 1.442695f; // ln2_recip
        vConstFloatPrgm1 = 2.0f;
        vConstFloatPrgm2 = 0.863281f;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_reciprocal_()
{
    vConstFloatPrgm0 = 1.442695f; // ln2_recip
    vConstFloatPrgm1 = 2.0f;
}

template <bool APPROXIMATION_MODE>
inline void _init_log_()
{
    vConstFloatPrgm0 = 0.692871f; // ln2

    // XXXXX could do these to higher precision
    vConstFloatPrgm1 = 0.1058f;
    vConstFloatPrgm2 = -0.7166f;
}

template <bool APPROXIMATION_MODE>
inline void _init_sqrt_()
{
    if (APPROXIMATION_MODE) {
        vConstFloatPrgm0 = s2vFloat16b(127 << 7);
    } else {
        vConstFloatPrgm0 = s2vFloat16b(0x5f37);
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
    _sfpu_load_imm16_(0, imm0);
    _sfpu_load_imm16_(1, imm1);
    _sfpu_load_imm16_(2, imm2);
}

template <bool APPROXIMATION_MODE>
inline void _init_sigmoid_()
{
    // imm0 = 0x3DFF;
    // imm1 = 0x21D8;
    // imm2 = 0xFF10;
    // TTI_SFPLOADI(0, 2, imm0);
    // TTI_SFPLOADI(1, 2, imm1);
    // TTI_SFPLOADI(2, 2, imm2);
    // Using a 6 piece LUT to calculate and model sigmoid  directly
    // x <= 0.5 --> 0.2452x + (-0.0004997)
    // x <= 1.0 --> 0.2173x + 0.0152
    // x <= 1.5 --> 0.1731x + 0.05988
    // x <= 2.0 --> 0.1262x + 0.1298
    // x <= 4.0 --> 0.0485x + 0.2998
    // x >  4.0 --> 0.4998

    // imm0[15:0] = A0=0.2452 = 0x33D9 -- imm0[31:16] = A1=0.2173 = 0x32F4
    _sfpu_load_imm32_(0,0x32F433D9);
    // imm4[15:0] = B0= -0.0004997  = 0x9018 -- imm4[31:16] = B1= 0.0152 = 0x23c8
    _sfpu_load_imm32_(4,0x23C89018);

    // imm1[15:0] = A2=0.1731 = 0x318a -- imm1[31:16] = A3=0.1262 = 0x300a
    _sfpu_load_imm32_(1,0x300A318A);
    // imm5[15:0] = B2=0.05988 = 0x2BAA -- imm5[31:16] = B3=0.1298 = 0x3027
    _sfpu_load_imm32_(5,0x30272BAA);

    // imm2[15:0] = A4=0.0485 = 0x2A35 -- imm2[31:16] = A5=0.0 = 0x7C00
    _sfpu_load_imm32_(2,0x7C002A35);
    // imm6[15:0] = B4=0.2998 = 0x34CC -- imm6[31:16] = B5=0.4998 = 0x37ff
    _sfpu_load_imm32_(6,0x37ff34CC);
}

template <bool APPROXIMATION_MODE>
inline void _init_gelu_derivative_()
{
    vConstFloatPrgm0 = 1.442695f; // ln2_recip
    vConstFloatPrgm1 = 2.0f;
    vConstFloatPrgm2 = 0.863281f;

    uint imm0;
    uint imm1;
    uint imm2;
    uint imm3;
    uint imm4;
    uint imm5;

    if constexpr (APPROXIMATION_MODE) {
        // Using a 6 piece LUT to calculate and model gelu_derivative directly
        // x <= 0.5 --> 0.8x + 0.5
        // x <= 1.0 --> 0.4x + 0.7
        // x <= 1.5 --> 0.1x + 0.99
        // x <= 2.0 --> -0.09x + 1.27
        // x <= 3.0 --> -0.075x + 1.235
        // x >  3.0 --> 1.0
        // imm0[15:0] = A0=0.8    = 0x3A66 -- imm0[31:16] = A1=0.4   = 0x3666
        imm0 = 0x36663A66;
        // imm1[15:0] = A2=0.1    = 0x2E66 -- imm1[31:16] = A3=-0.09 = 0xADC3
        imm1 = 0xADC32E66;
        // imm2[15:0] = A4=-0.075 = 0xACCD -- imm2[31:16] = A5=0     = 0x7C00
        imm2 = 0x7C00ACCD;
        // imm3[15:0] = B0=0.5    = 0x3800 -- imm3[31:16] = B1=0.7   = 0x399A
        imm3 = 0x399A3800;
        // imm4[15:0] = B2=0.99   = 0x3BEC -- imm4[31:16] = B3=1.27  = 0x3D14
        imm4 = 0x3D143BEC;
        // imm5[15:0] = B4=1.235  = 0x3CF1 -- imm5[31:16] = B5=1.0   = 0x3C00
        imm5 = 0x3C003CF1;
        _sfpu_load_imm32_(0, imm0);
        _sfpu_load_imm32_(1, imm1);
        _sfpu_load_imm32_(2, imm2);
        _sfpu_load_imm32_(4, imm3);
        _sfpu_load_imm32_(5, imm4);
        _sfpu_load_imm32_(6, imm5);
    } else {
        imm0 = 0x28FF;
        imm1 = 0x3020;
        _sfpu_load_imm16_(0, imm0);
        _sfpu_load_imm16_(1, imm1);
    }

}

template <bool APPROXIMATION_MODE>
inline void _init_gelu_()
{
    vConstFloatPrgm0 = 0.5f;

    // // >= 3.0f
    // lreg2_hi=0.50;//3800
    // lreg6_hi=0.0f;//7c00
    // // 2.0f -> 3.0f
    // lreg2_lo= 0.5402f;//3852
    // lreg6_lo= -0.1194f;//AFA4
    // // 1.5f -> 2.0f
    // lreg1_hi= .6099f; //38E1
    // lreg5_hi= -.2635f; //B437
    // // 1.0f -> 1.5f
    // lreg1_lo=0.6189;//38F3
    // lreg5_lo=-.2797;//B479
    // // 0.5f -> 1.0f
    // lreg0_hi=.4939f;//37E7
    // lreg4_hi=-.1605f;//B122
    // // 0.0f -> 0.5f
    // lreg0_lo=0.1928f;//322B
    // lreg4_lo=-0.0150f;//A3AE
    _sfpu_load_imm32_(0,0x37E7322B);
    _sfpu_load_imm32_(4,0xB12286D8);

    _sfpu_load_imm32_(1,0x38E138F3);
    _sfpu_load_imm32_(5,0xB437B479);

    _sfpu_load_imm32_(2,0x38003852);
    _sfpu_load_imm32_(6,0x7c00afa4);

}

inline void _init_dropout_(const uint seed)
{
    vConstIntPrgm0 = 0xb400;
    vConstIntPrgm1 = 0x1; // binary 0b1 - used to extract LSB

    _init_dropout_seed_(seed);
}

inline void _init_topk()
{
    _sfpu_load_config32_(0xF, 0x0, 0x4);          // Set bit [2] of the SFPU_CONTROL_REG to enable index tracking mode
}

inline void init_quant_zero_point(const uint zero_point)
{
    _sfpu_load_imm32_(2,zero_point);
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat _calculate_exponential_body_(vFloat in)
{
    vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        constexpr int FRAC_BITS = 3;
        constexpr uint SP_BIAS = 127 << FRAC_BITS;

        // * by 1/ln2 and add convert to 7.3 FxP format
        vFloat vConstLn2Recip = vConstFloatPrgm0;
        vFloat conv = in * vConstLn2Recip;

        // Clear exp bits
        vInt c23_73 = p_exp::C23_73;
        vInt tmp = reinterpret<vInt>(conv) - c23_73;

        // Add bias
        tmp += SP_BIAS;

        // SHL to move integer bits to exponent
        out = reinterpret<vFloat>(tmp << (10 - FRAC_BITS));
    }
    else
    {
        // Force sign to 0 (make number positive)
        out = _sfpu_exp_(setsgn(in, 0));

        v_if (in < 0) {
            out = _sfpu_reciprocal_(out);
        }
        v_endif;
    }

    return out;
}

/*
template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN>
void calculate_cube(uint16_t exp_base_scale_factor = 0)
{
    for (int d = 0; d < ITERATIONS; d++)
    {

        TTI_SFPLOAD(p_sfpu::LREG3, 0, 0); // load from dest
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPNOP; TTI_SFPNOP;
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);
        TTI_SFPNOP; TTI_SFPNOP;
        TTI_SFPSTORE(p_sfpu::LREG2, 0, 0); // Store from lreg[1] into dest registers
        TTI_INCRWC(0, 2, 0, 0);
    }
}
*/

template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS>
void _calculate_exponential_(const int iterations, uint16_t exp_base_scale_factor = 0)
{
    // Unroll 8 best for approx, unroll 0 for precise, compiler figures this out
    for (int d = 0; d < iterations; d++)
    {
        vFloat val = dst_reg[0];
        if constexpr(SCALE_EN){
            val = val * s2vFloat16a(exp_base_scale_factor);
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
                vFloat vConstLn2Recip = vConstFloatPrgm0;
                vFloat c23_73 = vConstFloatPrgm1;
                vInt adj_exp = vConstIntPrgm2;
                val = val * vConstLn2Recip + c23_73;

                // Remove Exponent of 7 and bias the Mantissa to 127.
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
            vFloat result = _sfpu_exp_(setsgn(val, 0));

            v_if (val < 0) {
                result = _sfpu_reciprocal_(result);
            }
            v_endif;

            dst_reg[0] = result;
        }

        dst_reg++;
    }
}
template <bool APPROXIMATION_MODE>
inline vFloat _calculate_gelu_core_(vFloat in)
{
    // SFPU microcode:
    // result = (APPROX_MODE == 1)
    //   ? (1 + erf(x/sqrt(2)))
    //   : (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) )
    vFloat result;
    if constexpr (APPROXIMATION_MODE) {
        result = in;
    } else {
        // f = (0.044715*x^3 + x)
        result = (in * in) * (in * s2vFloat16b(0.044715f)) + in;
        result *= s2vFloat16b(0.79788f);
    }

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_gelu_(const int iterations)
{

    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];
    vUInt l4 = l_reg[LRegs::LReg4];
    vUInt l5 = l_reg[LRegs::LReg5];
    vUInt l6 = l_reg[LRegs::LReg6];

    #pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        // vFloat in = dst_reg[0];
        // vFloat result = calculate_gelu_core<APPROXIMATION_MODE>(in);

        // vFloat half_in = in * half;
        // result = lut(result, l0, l1, l2);
        // result = half_in * result + half_in;

        //dst_reg[0] = result;

        vFloat in = dst_reg[0];
        vFloat half = vConstFloatPrgm0;
        vFloat half_in = in * half;
        vFloat result = lut2_sign(in, l0, l1, l2, l4, l5, l6);
        result = half_in + result;

        dst_reg[0] = result;

        dst_reg++;

        // dst_reg++;
        //TTI_SFPLOAD(3, 0, 1/*load addr mode*/,0);    // load from dest
        ////TTI_SFPMUL(3,11,9,7,0);           // lreg7 = 0.5*lreg3
        //TTI_SFPLUTFP32(7, 2);                // lreg7= LUT(3)
        //TTI_SFPMAD(3,12,7,3,0);            // lreg3 = 0.5*lreg3+lregm7
        //TTI_SFPSTORE(3, 0, 7/*store_addr_mod3*/, 0);   // and INCRWC by 4 using mode 3
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
    l_reg[LRegs::LReg4] = l4;
    l_reg[LRegs::LReg5] = l5;
    l_reg[LRegs::LReg6] = l6;


}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sigmoid_(const int iterations)
{
    constexpr int lut_mode = 0; // SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];
    vUInt l4 = l_reg[LRegs::LReg4];
    vUInt l5 = l_reg[LRegs::LReg5];
    vUInt l6 = l_reg[LRegs::LReg6];


    #pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        vFloat val = dst_reg[0];

        dst_reg[0] = lut2(val, l0, l1, l2, l4, l5, l6, lut_mode) + 0.5f;

        dst_reg++;
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
    l_reg[LRegs::LReg4] = l4;
    l_reg[LRegs::LReg5] = l5;
    l_reg[LRegs::LReg6] = l6;

}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_tanh_(const int iterations)
{
    // SFPU microcode
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    #pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
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
inline void _calculate_hardtanh_(const int iterations, uint param0, uint param1, uint param2)
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
    for (int d = 0; d < iterations; d++)
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
inline void _calculate_tanh_derivative_(const int iterations)
{
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    // tanh'(x) = 1 - (tanh(x))^2
    for (int d = 0; d < iterations; d++)
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
inline void _calculate_gelu_derivative_(const int iterations)
{
    if constexpr (APPROXIMATION_MODE) {
        constexpr int lut_mode = 1; // SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1

        vUInt l0 = l_reg[LRegs::LReg0];
        vUInt l1 = l_reg[LRegs::LReg1];
        vUInt l2 = l_reg[LRegs::LReg2];
        vUInt l4 = l_reg[LRegs::LReg4];
        vUInt l5 = l_reg[LRegs::LReg5];
        vUInt l6 = l_reg[LRegs::LReg6];

        // SFPU microcode:
        #pragma GCC unroll 0
        for (int d = 0; d < iterations; d++)
        {
            vFloat val = dst_reg[0];
            val = lut2(val, l0, l1, l2, l4, l5, l6, lut_mode);
            v_if (val < 0.0F) {
                val = val + 1.0f;
            }
            v_endif;
            dst_reg[0] = val;
            dst_reg++;

        }

        l_reg[LRegs::LReg0] = l0;
        l_reg[LRegs::LReg1] = l1;
        l_reg[LRegs::LReg2] = l2;
        l_reg[LRegs::LReg4] = l4;
        l_reg[LRegs::LReg5] = l5;
        l_reg[LRegs::LReg6] = l6;
    } else {
        constexpr uint imm2 = 0xFF10;

        vUInt l0 = l_reg[LRegs::LReg0];
        vUInt l1 = l_reg[LRegs::LReg1];

        // SFPU microcode:
        #pragma GCC unroll 0
        for (int d = 0; d < iterations; d++)
        {
            vFloat in = dst_reg[0];
            vFloat neg_half_sq_in = in * in * -0.5f;

            // exp = e^(val)
            vFloat exp = _calculate_exponential_body_<false>(neg_half_sq_in);

            // exp = exp * 1/sqrt(2*pi)
            vFloat partial = exp * in * s2vFloat16b(0.3989423F);

            vFloat result = _calculate_gelu_core_<true>(in);

            result = lut(result, l0, l1, imm2);

            dst_reg[0] = partial + result + 0.5f;
            dst_reg++;
        }

        l_reg[LRegs::LReg0] = l0;
        l_reg[LRegs::LReg1] = l1;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_reciprocal_(const int iterations)
{
    #pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        vFloat in = dst_reg[0];
        vFloat out = _sfpu_reciprocal_<APPROXIMATION_MODE ? 2 : 3>(in);

        v_if (in < 0.0F) {
            // Invert sign on calculated value if CC=1 (number is negative)
            out = -out;
        }
        v_endif;

        dst_reg[0] = out;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, int RECIPROCAL_ITERATIONS>
inline void _calculate_sqrt_(const int iterations)
{
    #pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        vFloat val = dst_reg[0];

        if constexpr (APPROXIMATION_MODE)
        {
            vUInt magic = vConstIntPrgm0;

            //sqrt initial approximation
            // adjust bias
            vUInt val_s = magic + reinterpret<vUInt>(val);

            // approximation of square root
            val_s >>= 1;
            dst_reg[0] = reinterpret<vFloat>(val_s);
        }
        else
        {
            // Recip root method
            //// Init approx
            //u.i = SQRT_MAGIC_F - (u.i >> 1);
            v_if (val != 0.0f)
            {
                vUInt magic = vConstIntPrgm0;
                vFloat approx = reinterpret<vFloat>(magic - (reinterpret<vUInt>(val) >> 1));

                //Reciproot iterations
                for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
                {
                    //x*r*(1.5f - xhalf*r*r);
                    approx = ((approx * approx) * (val * -0.5f) + 1.5f) * approx;
                }

                dst_reg[0] = approx * val;
            }
            v_endif;
        }

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_dropout_(const int iterations, uint prob, uint scale)
{
    // SFPU microcode

    FWLOG1("calculate_dropout() -- prob:%x", prob);
    FWLOG1("calculate_dropout() -- scale:%x", scale);

    vUInt rand = l_reg[LRegs::LReg3];

    #pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) {
        ////////////////////////
        // Scale samples
        ///////////////////////
        dst_reg[0] = dst_reg[0] * s2vFloat16b(scale);

        ////////////////////////
        // Drop samples
        ///////////////////////
        v_if (rand < prob) {
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

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_lrelu_(const int iterations, uint slope)
{
    // SFPU microcode
    vFloat s = s2vFloat16b(slope);

    #pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) {
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
inline void _calculate_power_(const int iterations, uint exponent)
{
    for (int d = 0; d < iterations; d++)
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
inline void _calculate_square_(const int iterations)
{
    #pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        vFloat in = dst_reg[0];
        vFloat result = in * in;

        dst_reg[0] = result;

        dst_reg++;
    }
}

template <bool HAS_BASE_SCALING>
sfpi_inline void _calculate_log_body_(const uint log_base_scale_factor)
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
inline void _calculate_log_(const int iterations, uint log_base_scale_factor)
{
    #pragma GCC unroll 8
    for(int d = 0; d < iterations; d++){
        _calculate_log_body_<HAS_BASE_SCALING>(log_base_scale_factor);
        dst_reg++;
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
inline void _calculate_comp_(const int iterations, uint exponent_size_8)
{

    // output_0 and output_1 hold the outputs use use when a zero or negative check is true/false.
    // False = 0.0 = kCONST_0 (5/8-bit exponent format)
    // True  = 1.0 = kCONST_1_FP16B (8-bit exponent format)
    // SFPU uses 8-bit exponent in operations so loading these constants in 8-bit exponent format.
    // Although a command flag can tell SFPU to re-bias a 5-bit exponent to 8-bit, we are loading 8-bit
    // exponent and telling SFPU to not add any bias to these constants.
    constexpr float output_0 = invert_output ? 0.0f : 1.0f;
    constexpr float output_1 = invert_output ? 1.0f : 0.0f;

    for (int d = 0; d < iterations; d++)
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
            // less_than_equal_zero
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
                // greater_than_zero
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

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_clamp_(const int iterations, uint param0, uint param1, uint param2)
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
    for (int d = 0; d < iterations; d++)
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
inline void _calculate_abs_(const int iterations)
{
    // SFPU microcode
    for (int d = 0; d < iterations; d++)
    {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sign_(const int iterations, uint exponent_size_8)
{
    // All params are in FP16 format
    // uint format = 1;
    #pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
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
inline void _calculate_max_(const int iterations)
{
    for (int d = 0; d < iterations; d++)
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
inline void _calculate_max_int32_(const int iterations)
{
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(2, 12, 3, 0);
        TTI_SFPLOAD(0, 12, 3, 64);
        TTI_SFPMOV(0, 0, 1, 0);
        TTI_SFPIADD(0, 2, 1, 2);
        TTI_SFPSTORE(0, 12, 3, 0);
        TTI_SFPENCC(0x003, 0, 0, 10);
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
inline void _calculate_sine_(const int iterations)
{
    // SFPU microcode
    for (int d = 0; d < iterations; d++)
    {
        vFloat v = dst_reg[0];
        v = 0.318309886183791f*v; // *1/pi to get number of pi rads.
        vInt whole_v = float_to_int16(v);
        vFloat whole_v_float = int32_to_float(whole_v, 0);
        v = v - whole_v_float;
        v *= 3.141592653589793f; // fractional * pi to get it in [-pi:pi]
        v = _sfpu_sine_maclaurin_series_<APPROXIMATION_MODE>(v);
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
inline void _calculate_cosine_(const int iterations)
{
    // SFPU microcode
    for (int d = 0; d < iterations; d++)
    {
        vFloat v = dst_reg[0];
        v = 0.318309886183791f*v; // *1/pi to get number of pi rads.
        vInt whole_v = float_to_int16(v);
        vFloat whole_v_float = int32_to_float(whole_v, 0);
        v = v - whole_v_float;
        v *= 3.141592653589793f; // fractional * pi to get it in [-pi:pi]
        v = _sfpu_cosine_maclaurin_series_<APPROXIMATION_MODE>(v);
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
inline void _relu_max_(const int iterations, uint uint_threshold)
{
    vFloat threshold = s2vFloat16(uint_threshold, s2vFloat16::fp16a);
    for (int d = 0; d < iterations; d++)
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
inline void _relu_min_(const int iterations, uint uint_threshold)
{
    vFloat threshold = s2vFloat16(uint_threshold, s2vFloat16::fp16a);
    for (int d = 0; d < iterations; d++)
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
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _cast_fp32_to_fp16a_(const int iterations)
{
    #pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
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
inline void _quant_int32_(const int iterations, const uint dst_offset)
{
    // Operand A is input (fp32)
    // Operand B is scaling factor (fp32)
    // Operand C is zero-point constant (fp32)
    // Output is int32 scaled to int8 range
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // operand A - fp32
        TTI_SFPLOAD(0, 3, 3, 0);
        // operand B - fp32 scaler
        TT_SFPLOAD(1, 3, 3, dst_offset * 64);
        // D(A) = A*B+C, LREG[2] = zero_point
        TTI_SFPMAD(0, 1, 2, 0, 0);
        // MAD has a 2-cycle pipeline latency so we need one cycle latency until next instr can consume the result
        TTI_NOP;
        // fp32->int8, descale value is zero (LREG_9)
        TTI_SFP_STOCH_RND(0,0,9,0,0,3);
        // LREG_0 -> dest as int32
        TTI_SFPSTORE(0,4,3,0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _requant_int32_(const int iterations, const uint dst_offset)
{
    // Operand A is input to requant (int32)
    // Operand B is scaling factor (fp32)
    // Operand C is zero-point constant (fp32)
    // Output is int32 scaled to int8 range
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // operand A - int32
        TTI_SFPLOAD(0, 4, 3, 0);
        // operand B - fp32 scaler
        TT_SFPLOAD(1, 3, 3, dst_offset*64);
        // cast int32->fp32
        TTI_SFPCAST(0, 0, 0);
        // D(A) = A*B+C, LREG[2] = zero_point
        TTI_SFPMAD(0, 1, 2, 0, 0);
        // MAD has a 2-cycle pipeline latency so we need one cycle latency until next instr can consume the result
        TTI_NOP;
        // fp32->int8, descale value is zero (LREG_9)
        TTI_SFP_STOCH_RND(0,0,9,0,0,3);
        // LREG_0 -> dest as int32
        TTI_SFPSTORE(0,4,3,0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _dequant_int32_(const int iterations, const uint dst_offset)
{
    // Operand A[LREG0] is input to dequant (int32)
    // Operand B[LREG1] is scaling factor (fp32)
    // Operand C[LREG2] is zero-point constant (fp32)
    // Output = (A + (-C)) * B (fp32)
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // operand A - int32
        TTI_SFPLOAD(0, 4, 3, 0);
        // operand B - fp32 scaler
        TT_SFPLOAD(1, 3, 3, dst_offset*64);
        // cast int32->fp32
        TTI_SFPCAST(0, 0, 0);
        // D(A)) = A+(-C), LREG[10] is 1, SFPADD = LREG_A*LREG_B+LREG_C
        TTI_SFPADD(0,10,2,0,0);
        TTI_NOP;
        // D(A)) = (A+(-C))*B, LREG[9] is zero
        TTI_SFPMUL(0,1,9,0,0);
        TTI_NOP;
        // LREG_0 -> dest as fp32
        TTI_SFPSTORE(0,3,3,0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _add_int32_(const int iterations, const uint dst_offset) {
    // Operand A is input1 (int32)
    // Operand B is input2 (int32)
    // Output is int32
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // operand A - int32
        TTI_SFPLOAD(0, 12, 3, 0);
        // operand B - int32
        TT_SFPLOAD(1, 12, 3, dst_offset * 64);
        TTI_SFPIADD(0, 1, 0, 4);
        // MAD has a 2-cycle pipeline latency so we need one cycle latency until next instr can consume the result
        TTI_NOP;
        // LREG_0 -> dest as int32
        TTI_SFPSTORE(0, 12, 3, 0);
        dst_reg++;
    }
}

inline void bitonic_topk_load8(uint offset, uint dist) {

    constexpr uint dst_indices_offset = 128;        // 2 tile x 64 rows per tile

    uint face_offset = offset >> 4;
    uint ld_offset = (offset & 0xF) + face_offset*32;

    // Load 16 consecutive numbers
    TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, ld_offset);
    TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, ld_offset + dist);
    
     // Load 16 consecutive indices
    TT_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_3, dst_indices_offset + ld_offset);            // How to load indices ? This is unpacked directly to dest!
    TT_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_3, dst_indices_offset + ld_offset + dist);

}

inline void bitonic_topk_store8(uint offset, uint dist) {
    constexpr uint dst_indices_offset = 128;        // 2 tile x 64 rows per tile

    uint face_offset = offset >> 4;
    uint ld_offset = (offset & 0xF) + face_offset*32;

    // Load 16 consecutive numbers
    TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, ld_offset);
    TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, ld_offset + dist);
    
     // Load 16 consecutive indices
    TT_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_3, dst_indices_offset + ld_offset + 0);      // How to load indices ? This is unpacked directly to dest!
    TT_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_3, dst_indices_offset + ld_offset + dist);

}

inline void bitonic_topk_load16(uint dist0, uint dist1) {
    constexpr uint dst_indices_offset = 128;        // 2 tile x 64 rows per tile

    // Load 16 consecutive numbers
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 4);
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, 8);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, 12);
    } else {
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 0 + dist0);
        TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, dist1);
        TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_3, dist1 + dist0);
    }
    
     // Load 16 consecutive indices
    TTI_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_3, dst_indices_offset + 0);      // How to load indices ? This is unpacked directly to dest!
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_3, dst_indices_offset + 4);
        TTI_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_3, dst_indices_offset + 8);
        TTI_SFPLOAD(p_sfpu::LREG7, 0, ADDR_MOD_3, dst_indices_offset + 12);
    } else {
        TT_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_3, dst_indices_offset + 0 + dist0);
        TT_SFPLOAD(p_sfpu::LREG6, 0, ADDR_MOD_3, dst_indices_offset + dist1);
        TT_SFPLOAD(p_sfpu::LREG7, 0, ADDR_MOD_3, dst_indices_offset + dist1 + dist0);
    }
}

inline void bitonic_topk_store16(uint dist0, uint dist1) {
    constexpr uint dst_indices_offset = 128;        // 2 tile x 64 rows per tile

    // Load 16 consecutive numbers
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 4);
        TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_3, 8);
        TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_3, 12);
    } else {
        TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0 + dist0);
        TT_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_3, dist1);
        TT_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_3, dist1 + dist0);
    }
    
     // Load 16 consecutive indices
    TTI_SFPSTORE(p_sfpu::LREG4, 0, ADDR_MOD_3, dst_indices_offset + 0);      // How to load indices ? This is unpacked directly to dest!
    if ((dist0 == 4) && (dist1 == 8)) {
        TTI_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_3, dst_indices_offset + 4);
        TTI_SFPSTORE(p_sfpu::LREG6, 0, ADDR_MOD_3, dst_indices_offset + 8);
        TTI_SFPSTORE(p_sfpu::LREG7, 0, ADDR_MOD_3, dst_indices_offset + 12);
    } else {
        TT_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_3, dst_indices_offset + 0 + dist0);
        TT_SFPSTORE(p_sfpu::LREG6, 0, ADDR_MOD_3, dst_indices_offset + dist1);
        TT_SFPSTORE(p_sfpu::LREG7, 0, ADDR_MOD_3, dst_indices_offset + dist1 + dist0);
    }

}

inline void bitonic_topk_ph3_st4_to_1(bool dir) {
    
    if (dir == (bool)SortDir::ArgMin) {
        TT_LOG("Issue max/min reverse");
        TTI_SFPCONFIG(0x104, 0xF, 1);      // Reverse the max/min behaviour of SWAP
        TTI_SFPNOP;
        TTI_SFPNOP;
    }

    // Step 4
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPNOP;

    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPNOP;

    TTI_SFPTRANSP(0,0,0,0);
    
    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPNOP;

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPNOP;

    TTI_SFPTRANSP(0,0,0,0);

    if (dir == (bool)SortDir::ArgMin) {
        TT_LOG("Restore max/min reverse");
        TTI_SFPCONFIG(0x004, 0xF, 1);      // Restore the max/min behaviour of SWAP
        TTI_SFPNOP;
        TTI_SFPNOP;
    }

}
inline void bitonic_topk_ph2_st3_to_1() {
    
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPNOP;

    TTI_SFPTRANSP(0,0,0,0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
    TTI_SFPNOP;

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);
    TTI_SFPNOP;

    TTI_SFPTRANSP(0,0,0,0);

}
inline void bitonic_topk_ph1_st2_to_1() {
    
    TTI_SFPTRANSP(0,0,0,0);

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);
    TTI_SFPNOP;

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);
    TTI_SFPNOP;

    TTI_SFPTRANSP(0,0,0,0);

}
inline void bitonic_topk_ph0_st1_to_1() {
    
    TTI_SFPTRANSP(0,0,0,0);
    
    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);
    TTI_SFPNOP;
    
    TTI_SFPTRANSP(0,0,0,0);
}

inline void bitonic_topk_step_N(bool dir) {
    // Step N
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    if (dir == (bool)SortDir::ArgMin) {
        // Min
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::UNCONDITIONALLY);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);
    }
    // MT: Maybe there's a way to optimize out unconditional transpose at every step. Think about this
}

inline void bitonic_topk_inc_x8_dest(uint inc, bool cr) {
    uint inc_grp8 = inc >> 3;
    for (uint i=0; i<inc_grp8; i++) {
        if (cr) {
            TTI_INCRWC(0b100, 8, 0, 0);
        } else {
            TTI_INCRWC(0, 8, 0, 0);
        }
    }
}

inline void bitonic_topk_inc_x4_dest(uint inc, bool cr) {
    uint inc_grp4 = inc >> 2;
    for (uint i=0; i<inc_grp4; i++) {
        if (cr) {
            TTI_INCRWC(0b100, 4, 0, 0);
        } else {
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _bitonic_topk_phases_steps(
    const int idir, const int i_end_phase, const int i_start_phase, const int i_end_step, const int i_start_step)
{
    // If more than 1 phase is requested, do all the steps from all phases
    // If 1 phase is requested, use i_start_step/i_end_step parameters
    uint dst_addr_offset = 0;
    for (int face=0; face<2; face++) {
        for (int col=0; col<2; col++) {
        
            bool dir = idir;
            for (int ph=i_start_phase; ph<(i_end_phase+1); ph++) {
                
                TT_LOG("Local Sort: phase = {}, dir = {}, idir = {}", ph, dir, idir);
                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                switch (ph) {
                case 0:
                    for (int d=0; d<4; d++) {
                        // Groups of 16 datums being sorted at the same time
                        bitonic_topk_load16(4, 8);
                        bitonic_topk_ph0_st1_to_1();
                        bitonic_topk_store16(4, 8);
                        TTI_INCRWC(0, 8, 0, 0);         // dst += 16
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                    }
                    break;
                case 1:
                    for (int d=0; d<4; d++) {
                        // Groups of 16 datums being sorted at the same time
                        bitonic_topk_load16(4, 8);
                        bitonic_topk_ph1_st2_to_1();
                        bitonic_topk_store16(4, 8);
                        TTI_INCRWC(0, 8, 0, 0);         // dst += 32
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                    }
                    break;
                case 2:
                    for (int d=0; d<4; d++) {
                        bitonic_topk_load16(4, 8);
                        bitonic_topk_ph2_st3_to_1();
                        bitonic_topk_store16(4, 8);
                        TTI_INCRWC(0, 8, 0, 0);         // dst += 32
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                    }
                    break;
                case 3:
                    for (int d=0; d<4; d++) {
                        bitonic_topk_load16(4, 8);
                        bitonic_topk_ph3_st4_to_1(dir);
                        bitonic_topk_store16(4, 8);
                        TTI_INCRWC(0, 8, 0, 0);         // dst += 32
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        TTI_INCRWC(0, 8, 0, 0);
                        dir = !dir;
                    }
                    break;
                default:
                    uint num_steps = ph+1;
                    uint start_step = (i_start_phase == i_end_phase) ? i_start_step : num_steps;
                    uint end_step = (i_start_phase == i_end_phase) ? i_end_step : 4;
                    uint sorted_seq_length = 1 << num_steps;
                    uint datums_compared = 0;
                    uint total_datums_to_compare = 64;
                    for (uint ss=start_step; ss>end_step; ss--) {
                        // Steps N to 5
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                        dir = idir;
                        uint dist = (ss == 5) ? 16 : 32;
                        uint inner_d = dist >> 3;                               // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
                        datums_compared = 0;
                        uint dst_offset = 0;
                        while (datums_compared < total_datums_to_compare) {
                            for (uint ii=0; ii<inner_d; ii++) {
                                bitonic_topk_load16(4, 2*dist);                 // load/store with offset of face 1 (in row major face layout)
                                bitonic_topk_step_N(dir);
                                bitonic_topk_store16(4, 2*dist);                // load/store with offset of face 1 (in row major face layout)
                                uint dst_inc = 8;
                                dst_offset += dst_inc;
                                bool dst_cr = false;
                                if (ii == (inner_d-1)) {
                                    dst_cr = true;
                                    dst_inc = 4*dist;
                                    dst_offset = 2*dist;
                                } else if (dst_offset == 16) {
                                    dst_cr = true;
                                    dst_inc = 32;
                                }
                                bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                datums_compared += 16;
                            }
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                        }
                    }
                    // steps 4 to 1
                    dir = idir;
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                    datums_compared = 0;
                    while (datums_compared < total_datums_to_compare) {
                        bitonic_topk_load16(4, 8);
                        bitonic_topk_ph3_st4_to_1(dir);
                        bitonic_topk_store16(4, 8);
                        bitonic_topk_inc_x8_dest(32, false);
                        datums_compared += 16;
                        dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                    }

                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
        
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _bitonic_topk_merge(const int m_iter, const int k)
{
    uint dst_addr_offset = 0;
    for (int face=0; face<2; face++) {
        for (int col=0; col<2; col++) {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            int k_max = k > 32 ? 32 : k;
            uint inner_d = k_max >> 2;                                                             // inner loop comparisons to sort len=K sequence;
            uint total_datums_to_compare = ((64 >> m_iter) < 2*k_max) ? 2*k_max : (64 >> m_iter);  // max(2, max(64, 64/(2^m))) total datums to compare; there's always at least 2*K datums
            uint dist = (k_max << m_iter) > 32 ? 32 : (k_max << m_iter);                           // min(32, k*2^k)
            uint ld_dist = (dist < 16) ? dist : 2*dist;                                            // Accounts for face offsets within a tile
            uint datums_compared = 0;
            uint dst_offset = 0;
            uint dst_cr = 0;
            TT_LOG("Merge - m = {}, k = {}, dist = {}, total_datums_to_compare = {}", m_iter, k, dist, total_datums_to_compare);
            while (datums_compared < total_datums_to_compare) {
                for (uint ii=0; ii<inner_d; ii++) {
                    bitonic_topk_load8(dst_offset, ld_dist);
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    bitonic_topk_store8(dst_offset, ld_dist);
                    datums_compared += 8;
                    if (ii == (inner_d-1)) {
                        dst_cr += 2*dist;
                        dst_offset = dst_cr;
                    } else {
                        dst_offset += 4;
                    }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _bitonic_topk_rebuild(const bool idir, const int m_iter, const int k, const int logk, const int skip_second)
{
    uint dst_addr_offset = 0;
    for (int face=0; face<2; face++) {
        for (int col=0; col<2; col++) {
            uint total_datums_shift = (skip_second&0x1);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
            uint rebuild_m = m_iter + 1;
            uint total_datums_to_compare = ((64 >> rebuild_m) < 2*k) ? 2*k : (64 >> rebuild_m);    // max(2*k, 64/(2^m)) total datums to compare; there's always at least 2*K datums
            total_datums_to_compare = total_datums_to_compare >> total_datums_shift;               // Reduce by 2 if skipping last 
            uint dist = (k << rebuild_m) > 32 ? 32 : (k << rebuild_m);                             // min(32, k*2^k)
            uint ld_offset = (dist >> 4)*32 + (dist & 0xF);
            uint ld_dist;
            int ph = logk-1;
            bool dir = idir;
            uint datums_compared = 0;
            TT_LOG("Rebuild - m = {}, k = {}, dist = {}, total_datums_to_compare = {}", m_iter, k, dist, total_datums_to_compare);
            switch (ph) {
                case 0:
                    TT_RISC_ASSERT(false, "K=2 not supported!");
                    break;
                case 1:
                    ld_dist = (ld_offset < 16) ? 4*ld_offset : 2*ld_offset;
                    while (datums_compared < total_datums_to_compare) {
                        // Groups of 16 datums being sorted at the same time
                        bitonic_topk_load16(ld_offset, ld_dist);
                        bitonic_topk_ph1_st2_to_1();
                        bitonic_topk_store16(ld_offset, ld_dist);
                        uint dst_inc = 2*32;
                        bitonic_topk_inc_x8_dest(dst_inc, false);
                        datums_compared += 16;
                    }
                    break;
                case 2:
                    while (datums_compared < total_datums_to_compare) {
                        // Groups of 16 datums being sorted at the same time
                        bitonic_topk_load16(4, ld_offset);
                        bitonic_topk_ph2_st3_to_1();
                        bitonic_topk_store16(4, ld_offset);
                        uint dst_inc = 2*32;
                        bitonic_topk_inc_x8_dest(dst_inc, false);
                        datums_compared += 16;
                    }
                    break;
                case 3:
                    while (datums_compared < total_datums_to_compare) {
                        // Groups of 16 datums being sorted at the same time
                        bitonic_topk_load16(4, 8);
                        bitonic_topk_ph3_st4_to_1(dir);
                        bitonic_topk_store16(4, 8);
                        uint dst_inc = 2*32;
                        bitonic_topk_inc_x8_dest(dst_inc, false);
                        datums_compared += 16;
                        dir = !dir;
                    }
                    break;
                default:
                    uint num_steps = ph+1;
                    uint start_step = num_steps;
                    uint end_step = 4;
                    uint sorted_seq_length = 1 << num_steps;
                    uint total_datums_to_compare = 64;
                    for (uint ss=start_step; ss>end_step; ss--) {
                        // Steps N to 5
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                        dir = idir;
                        datums_compared = 0;
                        uint dist = (ss == 5) ? 16 : 32; 
                        uint inner_d = dist >> 3;           // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
                        uint dst_offset = 0;
                        while (datums_compared < total_datums_to_compare) {
                            for (uint ii=0; ii<inner_d; ii++) {
                                bitonic_topk_load16(4, 2*dist);                 // load/store with offset of face 1 (in row major face layout)
                                bitonic_topk_step_N(dir);
                                bitonic_topk_store16(4, 2*dist);                // load/store with offset of face 1 (in row major face layout)
                                uint dst_inc = 8;
                                dst_offset += dst_inc;
                                bool dst_cr = false;
                                if (ii == (inner_d-1)) {
                                    dst_cr = true;
                                    dst_inc = 4*dist;
                                    dst_offset = 2*dist;
                                } else if (dst_offset == 16) {
                                    dst_cr = true;
                                    dst_inc = 32;
                                }
                                bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                datums_compared += 16;
                            }
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir;       // total_sorted = total_loops * 16; if total_sorted == sorted_seq_length
                        }
                    }
                    // steps 4 to 1
                    dir = idir;
                    datums_compared = 0;
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                    while (datums_compared < total_datums_to_compare) {
                        bitonic_topk_load16(4, 8);
                        bitonic_topk_ph3_st4_to_1(dir);
                        bitonic_topk_store16(4, 8);
                        bitonic_topk_inc_x8_dest(32, false);
                        datums_compared += 16;
                        dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                    }
                }
                
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);    
    }

}

} // namespace sfpu
} // namespace ckernel
