// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void sign_init() {
    // The paired store walks dest through ADDR_MOD_6, which advances by the two rows
    // the loop just wrote, so the loop body needs no separate increment.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sign(const uint /*exponent_size_8*/) {
    static_assert(ITERATIONS % 2 == 0, "calculate_sign() processes dest rows in pairs.");
    // A magnitude below the smallest NaN pattern is a number, so NaN and both zeros keep
    // 0 and everything else takes v's sign on 1.0. abs() == 0 also catches -0.0, which a
    // bare SFPSETCC compare can miss.
    const sfpi::vInt abs_mask(0x7fffffff);
    const sfpi::vInt nan_base(0x7f800001);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat res0 = 0.0f;
        sfpi::vInt mag_v0 = sfpi::as<sfpi::vInt>(v0) & abs_mask;
        v_if(sfpi::abs(v0) != 0.0F && (mag_v0 - nan_base) < 0) { res0 = sfpi::copysgn(sfpi::vFloat(1.0f), v0); }
        v_endif;
        sfpi::vFloat res1 = 0.0f;
        sfpi::vInt mag_v1 = sfpi::as<sfpi::vInt>(v1) & abs_mask;
        v_if(sfpi::abs(v1) != 0.0F && (mag_v1 - nan_base) < 0) { res1 = sfpi::copysgn(sfpi::vFloat(1.0f), v1); }
        v_endif;
        sfpi::dst_reg[0] = res0;
        sfpi::dst_reg[1].mode(ADDR_MOD_6) = res1;
    }
}

}  // namespace sfpu
}  // namespace ckernel
