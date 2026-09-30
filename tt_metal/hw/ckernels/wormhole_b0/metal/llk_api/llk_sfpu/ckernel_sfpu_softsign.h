// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // sfpi::approx_recip returns +-0 for any argument >= 2**126, so for
        // |x| >= 2**126 the reciprocal flushed to zero and softsign -- a function
        // bounded in (-1, 1) -- returned 0.0 where the answer is +-1.0.
        // Bound the ARGUMENT, never the result: for |x| >= 2**100 softsign(x) is
        // within 2**-100 of +-1, i.e. exact in fp32, and 1 + |x| then stays far
        // below the reciprocal's saturation point.
        v = sfpi::symmetric_clamp(v, 1.2676506e30f);  // 2**100
        sfpi::vFloat tmp = sfpi::abs(v) + 1.0f;
        tmp = sfpu_reciprocal<APPROXIMATION_MODE>(tmp);
        sfpi::dst_reg[0] = v * tmp;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
