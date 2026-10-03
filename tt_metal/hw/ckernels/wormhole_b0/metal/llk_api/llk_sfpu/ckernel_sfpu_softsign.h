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
        sfpi::vFloat tmp = sfpi::abs(v) + 1.0f;
        tmp = sfpu_reciprocal<APPROXIMATION_MODE>(tmp);
        sfpi::vFloat result = v * tmp;

        // 1/(1+|v|) is the intermediate that underflows. Once it drops below the smallest
        // normal it is flushed to zero and v * tmp collapses to 0 -- the opposite end of
        // softsign's range, where the exact result is +-1. ckernel_sfpu_gelu.h names the
        // same hazard ("Fused multiply avoids intermediate underflow") and restructures to
        // avoid it; softsign has no equivalent fused form, so guard the saturated region
        // instead. For |v| >= 2**26 the exact value of v/(1+|v|) already rounds to +-1.0f
        // in float32, so this is an exact substitution rather than a clamp. Non-finite
        // inputs skip the guard and keep their current behaviour.
        v_if(sfpi::is_finite(v) && sfpi::abs(v) >= 0x1.0p26f) {
            result = sfpi::copysgn(sfpi::vFloat(1.0f), v);
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
