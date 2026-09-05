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
        //
        // NaN is unchanged by this. Both forms dispatch on the sign bit -- SFPSETCC tests
        // it -- and a NaN reaching this kernel over the bf16 path has its sign bit clear,
        // so it takes the positive arm and yields 1.0 either way. Measured on device
        // before and after the rewrite; see test_vif_equiv_sweep's _NONFINITE_EXPECTED,
        // where the NaN column equals the +inf column for every one of these kernels.
        // (torch returns NaN here instead; neither the old nor the new kernel does, and
        // this rewrite does not change that.)
        vFloat r = sfpi::copysgn(vFloat(0.5f), v) + 0.5f;
        // SFPABS rather than a bare compare: SFPSETCC is unspecified for -0.0
        // (VectorUnit.md), so (v == 0.0f) can miss it and leave -0.0 at 0.0, where
        // -0.0 == 0 makes the answer s.
        v_if(sfpi::abs(v) == 0.0f) { r = s; }
        v_endif;

        dst_reg[0] = r;

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
