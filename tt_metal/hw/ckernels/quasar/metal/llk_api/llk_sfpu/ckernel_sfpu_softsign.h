// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_softsign() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // |v| clamped at 2**26 keeps 1 / (1 + |v|) normal; poison = v - v is NaN for +-inf and NaN inputs.
        // Subtracting poison keeps -0 as -0; the sign goes on before the multiply, so NaNs stay canonical.
        sfpi::vFloat va = sfpi::min(sfpi::setsgn(v, 0), sfpi::vFloat(sfpi::vConstFloatPrgm1));
        sfpi::vFloat denom = va + 1.0f;
        sfpi::vFloat poison = v - v;
        // With APPROXIMATION_MODE the compiler folds the subtraction into poison (computing v + (-v), +0 again) at the
        // cost of an instruction, so that mode keeps the add.
        sfpi::vFloat y = sfpi::copysgn(va, v) * _sfpu_reciprocal_<(APPROXIMATION_MODE) ? 0 : 2>(denom);
        if constexpr (APPROXIMATION_MODE) {
            sfpi::dst_reg[0].mode(ADDR_MOD_6) = y + poison;
        } else {
            sfpi::dst_reg[0].mode(ADDR_MOD_6) = y - poison;
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void init_softsign() {
    // The store walks dest through ADDR_MOD_6 (one SFPU row = two dest counter steps), so the loop needs no separate
    // increment.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    _init_reciprocal_<APPROXIMATION_MODE>();
    sfpi::vConstFloatPrgm1 = 0x1.0p26f;  // _sfpu_reciprocal_ only uses vConstFloatPrgm0 on Quasar
}

}  // namespace ckernel::sfpu
