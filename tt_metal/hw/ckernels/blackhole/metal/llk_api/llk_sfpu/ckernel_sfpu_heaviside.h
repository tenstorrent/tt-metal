// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void heaviside_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(uint value) {
    // SFPU microcode
    vFloat s = Converter::as_float(value);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        // SFPSETCC is unspecified for -0.0 (VectorUnit.md), so (v < 0.0f) can capture it
        // and report 0 where -0.0 == 0 makes the answer s. SFPABS clears the sign bit.
        // The trailing v_else still catches NaN, whose behaviour is unchanged.
        v_if(sfpi::abs(v) == 0.0f) { v = s; }
        v_elseif(v < 0.0f) { v = 0.0f; }
        v_elseif(v > 0.0f) { v = 1.0f; }
        v_else { v = s; }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
