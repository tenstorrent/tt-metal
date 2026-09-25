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
    // Unary SFPU sets addr_mod_base, so insn ADDR_MOD_2/3 map to physical 6/7.
    // Program physical slot 6 to advance dest by one row per SFPSTORE.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);

    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
    sfpi::vConstIntPrgm2 = 0xffff0000;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_square() {
    static_assert(ITERATIONS % 2 == 0, "calculate_square() processes dest rows in pairs.");

    // Two dest rows in flight; instructions alternate so each two-cycle SFPU
    // write is covered by the other row. On Wormhole, SFPSHFT mod 1 is in-place,
    // so copy the product before taking the RNE LSB.
    for (int d = 0; d < ITERATIONS; d += 2) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        if constexpr (!is_fp32_dest_acc_en) {
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
            TTI_SFPSHFT((-16) & 0xFFF, 0, p_sfpu::LREG2, 1);                            // lsb0 = bits0 >> 16
            TTI_SFPSHFT((-16) & 0xFFF, 0, p_sfpu::LREG3, 1);                            // lsb1 = bits1 >> 16
            TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG2, 0);                            // lsb0 &= 1
            TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG3, 0);                            // lsb1 &= 1
            TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);  // bits0 += 0x7fff
            TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);  // bits1 += 0x7fff
            TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);   // bits0 += lsb0
            TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);   // bits1 += lsb1
            TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG0, 0);                            // bits0 &= 0xffff0000
            TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG1, 0);                            // bits1 &= 0xffff0000
        }
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
    }
}

}  // namespace ckernel::sfpu
