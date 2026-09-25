// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

inline void square_init() {
    // The paired store walks dest through ADDR_MOD_6, which advances by the two rows
    // the loop just wrote (one sfpi row is two dest counter steps), so the loop body
    // needs no separate increment.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);

    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
    sfpi::vConstIntPrgm2 = 0xffff0000;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_square() {
    static_assert(ITERATIONS % 2 == 0, "calculate_square() processes dest rows in pairs.");

#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat r0 = v0 * v0;
        sfpi::vFloat r1 = v1 * v1;
        if constexpr (!is_fp32_dest_acc_en) {
            // SFPSTORE into a bf16 dest truncates; round to nearest-even first
            // so square stays bit-exact with ttnn.mul(x, x).
            sfpi::vUInt bits0 = sfpi::as<sfpi::vUInt>(r0);
            sfpi::vUInt bits1 = sfpi::as<sfpi::vUInt>(r1);
            sfpi::vUInt lsb0 = (bits0 >> 16) & sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm0);
            sfpi::vUInt lsb1 = (bits1 >> 16) & sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm0);
            bits0 = bits0 + sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm1) + lsb0;
            bits1 = bits1 + sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm1) + lsb1;
            bits0 = bits0 & sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm2);
            bits1 = bits1 & sfpi::as<sfpi::vUInt>(sfpi::vConstIntPrgm2);
            r0 = sfpi::as<sfpi::vFloat>(bits0);
            r1 = sfpi::as<sfpi::vFloat>(bits1);
        }
        sfpi::dst_reg[0] = r0;
        sfpi::dst_reg[1].mode(ADDR_MOD_6) = r1;
    }
}

}  // namespace ckernel::sfpu
