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
        // |v| clamped at 2**26 keeps 1 / (1 + |v|) normal; poison = v - v is NaN for +-inf and NaN inputs.
        // 2**26 is an immediate because sfpu_reciprocal uses all three programmable constants on Wormhole.
        sfpi::vFloat va = sfpi::min(sfpi::setsgn(v, 0), sfpi::vFloat(0x1.0p26f));
        sfpi::vFloat denom = va + 1.0f;
        sfpi::vFloat poison = v - v;
        sfpi::vFloat tmp = va * sfpu_reciprocal<APPROXIMATION_MODE>(denom) + poison;
        // Re-signing after the multiply keeps the sign of a zero result; a NaN result follows the input.
        // ADDR_MOD_2 here is ADDR_MOD_6 once the SFPU address-mode base is applied.
        sfpi::dst_reg[0].mode(ADDR_MOD_2) = sfpi::copysgn(tmp, v);
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    // The store walks dest through ADDR_MOD_6 (one sfpi row = two dest counter steps), so the loop needs no separate
    // increment.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
