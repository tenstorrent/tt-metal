/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_recip.h"
#include <limits>

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{
sfpi_inline vFloat sfpu_exp(vFloat val)
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

template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN=false, int ITERATIONS=4>
inline void calculate_exponential(int16_t exp_base_scale_factor = 0)
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

        if constexpr(SCALE_EN) {
            val = val * s2vFloat16a(exp_base_scale_factor);
            dst_reg[0] = val;
        }
        if constexpr (APPROXIMATION_MODE)
        {
            v_if (val >= 89) {
                vFloat val_inf = std::numeric_limits<float>::infinity();
                dst_reg[0] = val_inf;
            } v_elseif(val < -42) {
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
            val = sfpu_exp(setsgn(val, 0));

            vFloat orig = dst_reg[0];

            // Loaded by reciprocal
            dst_reg[0] = val;
            v_if (orig < 0) {
                dst_reg[0] = sfpu_reciprocal<false>(val);
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
void exp_init() {
    if constexpr(APPROXIMATION_MODE) {
        TTI_SFPLOADI(p_sfpu::LREG0, 0, p_exp::C23_73);
        TTI_SFPLOADI(p_sfpu::LREG2, 0, p_exp::ADJ_EXP);
    }
}

template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE>
sfpi_inline vFloat calculate_exponential_body(vFloat in)
{
    vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        v_if (in >= 89) {
            vFloat val_inf = std::numeric_limits<float>::infinity();
            out = val_inf;
        } v_elseif(in < -42) {
            out = 0.0f;
        } v_else {
            // * by 1/ln2 and add convert to 7.3 FxP format
            vFloat val = in * vConst1p4424 + p_exp::C23_73;

            // Remove Exponent of 7 and bias the Mantissa to 127.
            // LREG2 already holds 2's complement value so we simply do REG2 + REG3
            vInt val_short = p_exp::ADJ_EXP + reinterpret<vInt>(val);

            // SHL to move integer bits to exponent
            val_short <<= 10 - p_exp::FRAC_BITS;
            out = reinterpret<vFloat>(val_short);
        }
        v_endif;
    }
    else
    {
        // Force sign to 0 (make number positive)
        vFloat exp = sfpu_exp(setsgn(in, 0));

        // Load input value, to determine whether reciprocal needs to be run
        vFloat val = dst_reg[0];

        // store tentatively e^x
        // reciprocal function relies on reloading input
        dst_reg[0] = exp;

        v_if (val < 0) {
            dst_reg[0] = sfpu_reciprocal<true>(exp);
        }
        v_endif;
    }
    return out;
}


template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE>
sfpi_inline vFloat calculate_exponential_body_improved(vFloat in)
{
    vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        vInt adj_exp;
        adj_exp = l_reg[LRegs::LReg2];
        v_if (in >= 89) {
            vFloat val_inf = std::numeric_limits<float>::infinity();
            out = val_inf;
        } v_elseif(in < -42.0f) {
            out = 0.0f;
        } v_else {
            // * by 1/ln2 and add convert to 7.3 FxP format
            vFloat val = in * vConst1p4424 + s2vFloat16b(p_exp::C23_73);

            // Remove Exponent of 7 and bias the Mantissa to 127.
            // LREG2 already holds 2's complement value so we simply do REG2 + REG3
            vInt val_short =  adj_exp + reinterpret<vInt>(val);

            // SHL to move integer bits to exponent
            val_short <<= 10 - p_exp::FRAC_BITS;
            out = reinterpret<vFloat>(val_short);
        }
        v_endif;
        l_reg[LRegs::LReg2] = adj_exp;
    }
    else
    {
        // Force sign to 0 (make number positive)
        vFloat exp = sfpu_exp(setsgn(in, 0));

        // Load input value, to determine whether reciprocal needs to be run
        vFloat val = dst_reg[0];

        // store tentatively e^x
        // reciprocal function relies on reloading input
        out = exp;

        v_if (val < 0) {
            //dst_reg[0] = sfpu_reciprocal<true>(exp);
            out = sfpu_reciprocal<true>(exp);
        }
        v_endif;
    }
    return out;
}



} // namespace sfpu
} // namespace ckernel
