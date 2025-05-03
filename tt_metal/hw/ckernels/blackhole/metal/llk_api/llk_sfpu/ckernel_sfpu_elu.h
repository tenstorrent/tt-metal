// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_elu(uint slope) {
    // SFPU microcode
    Converter c_slope;
    c_slope.u = slope;
    vFloat s = c_slope.f;

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];

        v_if(v < 0.0f) {
            vFloat v_exp = calculate_exponential_body_improved<APPROXIMATION_MODE>(v);
            v = s * (v_exp - 1.0f);
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void elu_init() {
    if constexpr (APPROXIMATION_MODE) {
        vConstFloatPrgm0 = 1.442695f;  // ln2_recip
        vConstFloatPrgm1 = s2vFloat16b(p_exp::C23_73);
        vConstFloatPrgm2 = s2vFloat16b(p_exp::ADJ_EXP);
    } else {
        vConstFloatPrgm0 = 1.442695f;  // ln2_recip
        vConstFloatPrgm1 = 2.0f;
        vConstFloatPrgm2 = 0.863281f;
    }
}

}  // namespace sfpu
}  // namespace ckernel
