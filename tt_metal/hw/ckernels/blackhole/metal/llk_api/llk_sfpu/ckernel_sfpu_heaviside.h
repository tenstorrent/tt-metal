// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(uint value) {
    // SFPU microcode
    Converter c_value;
    c_value.u = value;
    vFloat s = c_value.f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        v_if(v < 0.0f) { v = 0.0f; }
        v_elseif(v > 0.0f) { v = 1.0f; }
        v_else { v = s; }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
