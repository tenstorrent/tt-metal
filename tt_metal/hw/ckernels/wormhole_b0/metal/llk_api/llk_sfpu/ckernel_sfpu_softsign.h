// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_softsign() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat va = sfpi::setsgn(v, 0);
        sfpi::vFloat in = va + 1.0f;
        sfpi::vFloat negative_x = sfpi::copyman(-1.0f, in);
        sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;
        sfpi::vUInt scale_bits = ~sfpi::as<sfpi::vUInt>(in);
        y = sfpi::vConstFloatPrgm2 + y * negative_x;
        sfpi::vFloat scale = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits), 0);
        sfpi::vFloat t = 1.0f + negative_x * y;
        y = y + y * t;
        if constexpr (!APPROXIMATION_MODE) {
            t = 1.0f + negative_x * y;
            y = y + y * t;
        }
        sfpi::vFloat r = (va * scale) * (y * 0.5f);
        sfpi::dst_reg[0].mode(ADDR_MOD_2) = sfpi::copysgn(r, v);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void init_softsign() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
