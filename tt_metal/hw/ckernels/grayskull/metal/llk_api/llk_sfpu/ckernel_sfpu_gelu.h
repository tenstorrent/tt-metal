// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"

#include "sfpi.h"
#include "ckernel_sfpu_cdf.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE>
void gelu_init() {
    _init_gelu_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
void gelu_derivative_init() {
    if constexpr(APPROXIMATION_MODE) {
        TTI_SFPLOADI(p_sfpu::LREG2, 0, p_exp::ADJ_EXP);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_gelu_appx()
{
    _calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>();
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

template <bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_gelu_derivative()
{
    _calculate_gelu_derivative_<APPROXIMATION_MODE, ITERATIONS>();
}

} // namespace sfpu
} // namespace ckernel
