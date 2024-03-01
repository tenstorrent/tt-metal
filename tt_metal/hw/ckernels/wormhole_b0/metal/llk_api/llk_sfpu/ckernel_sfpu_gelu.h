// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"
#include "ckernel_sfpu_cdf.h"
#include "ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <int ITERATIONS>
inline void calculate_gelu_appx()
{

    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];
    vUInt l4 = l_reg[LRegs::LReg4];
    vUInt l5 = l_reg[LRegs::LReg5];
    vUInt l6 = l_reg[LRegs::LReg6];

    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
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
        //TTI_SFPSTORE(3, 0, 3/*store_addr_mod3*/, 0);   // and INCRWC by 4 using mode 3
    }

    l_reg[LRegs::LReg0] = l0;
    l_reg[LRegs::LReg1] = l1;
    l_reg[LRegs::LReg2] = l2;
    l_reg[LRegs::LReg4] = l4;
    l_reg[LRegs::LReg5] = l5;
    l_reg[LRegs::LReg6] = l6;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu()
{
    if constexpr (APPROXIMATION_MODE) {
	    calculate_gelu_appx<ITERATIONS>();
    } else {
        constexpr bool scaled = true;
        // SFPU microcode
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat val = dst_reg[0];
            vFloat result = calculate_cdf_appx(val,scaled);
            dst_reg[0] = result;
            dst_reg++;
	    }
    }
}

template <bool APPROXIMATION_MODE>
void gelu_init() {
    vConstFloatPrgm0 = 0.5f;
    if constexpr (APPROXIMATION_MODE) {
        // // >= 3.0f
        // lreg2_hi=0.50;//3800
        // lreg6_hi=0.0f;//7c00
        // // 2.0f -> 3.0f
        // lreg2_lo= 0.537f;//384B
        // lreg6_lo= -0.116f;//AF6C
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
        // lreg4_lo=-0.00010443f;//86D8
        _sfpu_load_imm32_(0,0x37E7322B);
        //_sfpu_load_imm32_(4,0xB122A3AE);
        _sfpu_load_imm32_(4,0xB12286D8);


        _sfpu_load_imm32_(1,0x38E138F3);
        _sfpu_load_imm32_(5,0xB437B479);

        _sfpu_load_imm32_(2,0x3800384B);
        _sfpu_load_imm32_(6,0x7c00AF6C);
    }
}


template <bool APPROXIMATION_MODE>
void gelu_derivative_init() {
    vConstFloatPrgm0 = 1.442695f; // ln2_recip
    vConstFloatPrgm1 = 2.0f;
    vConstFloatPrgm2 = 0.863281f;
    if constexpr (APPROXIMATION_MODE) {
        uint imm0_high;
        uint imm0_low;
        uint imm1_high;
        uint imm1_low;
        uint imm2_high;
        uint imm2_low;
        uint imm3_high;
        uint imm3_low;
        uint imm4_high;
        uint imm4_low;
        uint imm5_high;
        uint imm5_low;
        // Using a 6 piece LUT to calculate and model gelu_derivative directly
        // x <= 0.5 --> 0.8x + 0.5
        // x <= 1.0 --> 0.4x + 0.7
        // x <= 1.5 --> 0.1x + 0.99
        // x <= 2.0 --> -0.09x + 1.27
        // x <= 3.0 --> -0.075x + 1.235
        // x >  3.0 --> 1.0
        // imm0[15:0] = A0=0.8    = 0x3A66 -- imm0[31:16] = A1=0.4   = 0x3666
        imm0_high = 0x3666;
        imm0_low  = 0x3A66;
        // imm1[15:0] = A2=0.1    = 0x2E66 -- imm1[31:16] = A3=-0.09 = 0xADC3
        imm1_high = 0xADC3;
        imm1_low  = 0x2E66;
        // imm2[15:0] = A4=-0.075 = 0xACCD -- imm2[31:16] = A5=0     = 0x7C00
        imm2_high = 0x7C00;
        imm2_low  = 0xACCD;
        // imm3[15:0] = B0=0.5    = 0x3800 -- imm3[31:16] = B1=0.7   = 0x399A
        imm3_high = 0x399A;
        imm3_low  = 0x3800;
        // imm4[15:0] = B2=0.99   = 0x3BEC -- imm4[31:16] = B3=1.27  = 0x3D14
        imm4_high = 0x3D14;
        imm4_low  = 0x3BEC;
        // imm5[15:0] = B4=1.235  = 0x3CF1 -- imm5[31:16] = B5=1.0   = 0x3C00
        imm5_high = 0x3C00;
        imm5_low  = 0x3CF1;
        TTI_SFPLOADI(0, 10, imm0_low);
        TTI_SFPLOADI(0,  8, imm0_high);
        TTI_SFPLOADI(1, 10, imm1_low);
        TTI_SFPLOADI(1,  8, imm1_high);
        TTI_SFPLOADI(2, 10, imm2_low);
        TTI_SFPLOADI(2,  8, imm2_high);
        TTI_SFPLOADI(4, 10, imm3_low);
        TTI_SFPLOADI(4,  8, imm3_high);
        TTI_SFPLOADI(5, 10, imm4_low);
        TTI_SFPLOADI(5,  8, imm4_high);
        TTI_SFPLOADI(6, 10, imm5_low);
        TTI_SFPLOADI(6,  8, imm5_high);
    }
    else {
        uint imm0;
        uint imm1;
        imm0 = 0x28FF;
        imm1 = 0x3020;
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
    }
}


template <bool APPROXIMATION_MODE>
inline vFloat calculate_gelu_core(vFloat in)
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

template <bool APPROXIMATION_MODE,int ITERATIONS = 8>
inline void calculate_gelu_derivative()
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
        for (int d = 0; d < ITERATIONS; d++)
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
        for (int d = 0; d < ITERATIONS; d++)
        {
            vFloat in = dst_reg[0];
            vFloat neg_half_sq_in = in * in * -0.5f;

            // exp = e^(val)
            vFloat exp = calculate_exponential_body<false,ITERATIONS>(neg_half_sq_in);

            // exp = exp * 1/sqrt(2*pi)
            vFloat partial = exp * in * s2vFloat16b(0.3989423F);

            vFloat result = calculate_gelu_core<true,ITERATIONS>(in);

            result = lut(result, l0, l1, imm2);

            dst_reg[0] = partial + result + 0.5f;
            dst_reg++;
        }

        l_reg[LRegs::LReg0] = l0;
        l_reg[LRegs::LReg1] = l1;
    }
}

} // namespace sfpu
} // namespace ckernel
