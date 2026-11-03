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
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat a = sfpi::abs(v);
        sfpi::vFloat res = a * sfpu_reciprocal<APPROXIMATION_MODE>(a + 1.0f);
        // For finite |v| >= 2^25 the result already rounds to 1, and the reciprocal would otherwise flush to 0 near
        // fp32 max. For finite v, a is non-negative and its bits order like its value; inf and NaN fall outside the
        // range and keep the formula's result.
        sfpi::vInt bits = sfpi::as<sfpi::vInt>(a);
        v_if(bits >= 0x4C000000 && bits < 0x7F800000) { res = 1.0f; }
        v_endif;
        sfpi::dst_reg[0] = sfpi::copysgn(res, v);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
