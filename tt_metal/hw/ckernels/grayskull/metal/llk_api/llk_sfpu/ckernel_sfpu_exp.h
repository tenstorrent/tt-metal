// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
    return _sfpu_exp_(val);
}

template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE = false, bool SCALE_EN=false, int ITERATIONS=4>
inline void calculate_exponential(int16_t exp_base_scale_factor = 0)
{
    _calculate_exponential_<APPROXIMATION_MODE, ZERO_NEGATIVE, SCALE_EN, ITERATIONS>(exp_base_scale_factor);
}


template <bool APPROXIMATION_MODE>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE = false>
sfpi_inline vFloat calculate_exponential_body(vFloat in)
{
    vFloat out;

    if constexpr (APPROXIMATION_MODE) {
        v_if (in >= 89) {
            vFloat val_inf = std::numeric_limits<float>::infinity();
            out = val_inf;
        } v_elseif(in < -42) {
            out = 0.0f;
        } v_else {
            out = _calculate_exponential_body_<APPROXIMATION_MODE, ZERO_NEGATIVE>(in);
        }
        v_endif;
    } else {
        out = _calculate_exponential_body_<APPROXIMATION_MODE, ZERO_NEGATIVE>(in);
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
