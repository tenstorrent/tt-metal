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

        vFloat r = sfpi::copysgn(vFloat(0.5f), v) + 0.5f;
        // SFPSETCC is unspecified for -0.0 (VectorUnit.md), so a bare v == 0.0f can miss it.
        v_if(sfpi::abs(v) == 0.0f) { r = s; }
        v_endif;

        dst_reg[0] = r;

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
