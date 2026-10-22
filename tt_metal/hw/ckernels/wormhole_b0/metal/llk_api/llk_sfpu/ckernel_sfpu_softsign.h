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
        sfpi::vFloat a = sfpi::abs(v);
        sfpi::vFloat r = a * sfpu_reciprocal<APPROXIMATION_MODE>(a + 1.0f);
        // 1/(1+|x|) underflows near FLT_MAX; for |x| >= 2^24 softsign rounds to 1 anyway
        v_if(a >= 16777216.0f) { r = 1.0f; }
        v_endif;
        // copysgn restores the sign, including for -0
        sfpi::dst_reg[0] = sfpi::copysgn(r, v);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
