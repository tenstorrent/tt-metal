/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_cdf.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_gelu_appx()
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

template <bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_gelu()
{

    if constexpr (APPROXIMATION_MODE) {
	    calculate_gelu_appx<APPROXIMATION_MODE,ITERATIONS>();
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
    uint imm0;
    uint imm1;
    uint imm2;
    imm0 = 0x18FF;
    imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
    imm2 = 0xFF00;
    TTI_SFPLOADI(0, 2, imm0);
    TTI_SFPLOADI(1, 2, imm1);
    TTI_SFPLOADI(2, 2, imm2);
}


template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_gelu_core(vFloat in)
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

template <bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_gelu_derivative()
{
    // SFPU microcode:
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat result = val * val * vConstNeg0p5;

        // exp = e^(val) * 1/sqrt(2*pi)
        if constexpr(APPROXIMATION_MODE) {
            vFloat exp = calculate_exponential_body_improved<APPROXIMATION_MODE, APPROXIMATION_MODE>(result);
            exp *= 0.39844F;
            dst_reg[0] = exp * val;
        } else {
            dst_reg[0] = result;
            vFloat exp = calculate_exponential_body_improved<APPROXIMATION_MODE, APPROXIMATION_MODE>(result);
            exp *= 0.39844F;
            dst_reg[0] = exp * val;
        }
        result = calculate_gelu_core<APPROXIMATION_MODE>(val);

        dst_reg[0] = dst_reg[0] + result;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void gelu_derivative_init() {
    if constexpr(APPROXIMATION_MODE) {
        TTI_SFPLOADI(p_sfpu::LREG2, 0, p_exp::ADJ_EXP);
    }
}

} // namespace sfpu
} // namespace ckernel
