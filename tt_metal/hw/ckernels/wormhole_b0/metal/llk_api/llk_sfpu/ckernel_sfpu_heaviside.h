// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void heaviside_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(std::uint32_t value) {
    vFloat s = Converter::as_float(value);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        // copysgn(0.5, v) + 0.5 is 1.0 for a clear sign bit and 0.0 for a set one, which
        // reproduces the v<0 and v>0 arms in two unpredicated slots. Only exact zero is
        // left to branch on, so the three-way chain's SFPPUSHC/SFPPOPC pair disappears.
        // Sign-bit dispatch also matches the old chain on NaN: SFPSETCC tests the sign
        // bit, so -NaN took the v<0 arm and +NaN took the v>0 arm, same as here.
        vFloat r = sfpi::copysgn(vFloat(0.5f), v) + 0.5f;
        v_if(v == 0.0f) { r = s; }
        v_endif;

        dst_reg[0] = r;

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
