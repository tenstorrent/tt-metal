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
        //
        // The bound is taken on |v| once and the sign restored afterwards. This is
        // symmetric_clamp(v) followed by abs(v) bit for bit -- the clamped value's
        // magnitude is exactly a and the product takes the sign of its first factor
        // because the reciprocal is positive -- without recomputing the magnitude
        // the bound already produced.
        const sfpi::vFloat a   = sfpi::min(sfpi::abs(v), 1.2676506e30f);  // 2**100
        const sfpi::vFloat tmp = sfpu_reciprocal<APPROXIMATION_MODE>(a + 1.0f);
        sfpi::dst_reg[0]       = sfpi::copysgn(a, v) * tmp;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
