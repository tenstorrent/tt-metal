// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {


template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_elu(uint slope)
{
    // SFPU microcode
    constexpr bool zero_negative = true;
    Converter c_slope;
    c_slope.u = slope;

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        v_if (v < 0.0f) {
            vFloat v_exp = calculate_exponential_body_improved<true, zero_negative>(v);
            vFloat s = c_slope.f;
            v = s*(v_exp - 1.0f);
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void elu_init() {
    // This should be for approx mode, but we run out of registers if we try
    // to run non-approx mode, so approx and non-approx mode will be the same for GS
    // For WH there are correct approx and non-approx modes
    TTI_SFPLOADI(p_sfpu::LREG2, 0, p_exp::ADJ_EXP);
}

}  // namespace sfpu
}  // namespace ckernel
