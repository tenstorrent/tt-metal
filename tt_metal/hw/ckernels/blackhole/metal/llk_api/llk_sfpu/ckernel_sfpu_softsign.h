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
        // Clamp |v| at 2**26 (vConstFloatPrgm1): from there on v / (1 + |v|) rounds to +-1.0f in float32, and the clamp
        // keeps 1 / (1 + |v|) far from the smallest normal, below which it flushes to zero. poison = v - v is NaN for
        // +-inf and NaN inputs and +0 otherwise, so those inputs still produce NaN. It is subtracted rather than added:
        // -0 - (+0) stays -0, while -0 + (+0) rounds to +0. The sign goes on before the multiply, so the result is
        // never re-signed and a NaN keeps the sign the hardware gives it.
        sfpi::vFloat va = sfpi::min(sfpi::setsgn(v, 0), sfpi::vFloat(sfpi::vConstFloatPrgm1));
        sfpi::vFloat denom = va + 1.0f;
        sfpi::vFloat poison = v - v;
        // With APPROXIMATION_MODE the compiler folds the subtraction into poison (computing v + (-v), +0 again) at the
        // cost of an instruction, so that mode keeps the add.
        sfpi::vFloat y = sfpi::copysgn(va, v) * sfpu_reciprocal<APPROXIMATION_MODE>(denom);
        if constexpr (APPROXIMATION_MODE) {
            sfpi::dst_reg[0].mode(ADDR_MOD_6) = y + poison;
        } else {
            sfpi::dst_reg[0].mode(ADDR_MOD_6) = y - poison;
        }
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    // The store walks dest through ADDR_MOD_6 (one sfpi row = two dest counter steps), so the loop needs no separate
    // increment.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
    sfpi::vConstFloatPrgm1 = 0x1.0p26f;  // sfpu_reciprocal only uses vConstFloatPrgm0 on Blackhole
}

}  // namespace ckernel::sfpu
