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
    // ADDR_MOD_6 advances dest by one row per SFPSTORE; the loop stores twice per iteration.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // RNE constants, held in LREG12/13/14 for the whole kernel.
    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
    sfpi::vConstIntPrgm2 = 0xffff0000;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_square() {
    static_assert(ITERATIONS % 2 == 0, "calculate_square() processes dest rows in pairs.");

    // Two dest rows are in flight at once (LREG0/LREG1, with LREG2/LREG3 holding their
    // tie-break bits), and the instructions for the two rows alternate so that each
    // two-cycle SFPU write is covered by the other row's independent instruction.
    // sfpi emits the same opcodes but is free to reorder them onto one register, which
    // leaves the rounding chain serial; issuing them here pins the interleaving.
    // Four loop trips (ITERATIONS=8, d+=2); GCC fully unrolls this without a pragma.
    for (int d = 0; d < ITERATIONS; d += 2) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 2);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        if constexpr (!is_fp32_dest_acc_en) {
            // float32_to_bf16_rne() expanded over both rows: SFPSTORE into a bf16 dest
            // truncates, so round to nearest-even first to stay bit-exact with ttnn.mul.
            TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG2, 5);                // lsb0 = bits0 >> 16
            TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG1, p_sfpu::LREG3, 5);                // lsb1 = bits1 >> 16
            TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG2, 0);                            // lsb0 &= 1
            TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG3, 0);                            // lsb1 &= 1
            TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);  // bits0 += 0x7fff
            TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);  // bits1 += 0x7fff
            TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);   // bits0 += lsb0
            TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);   // bits1 += lsb1
            TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG0, 0);                            // bits0 &= 0xffff0000
            TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG1, 0);                            // bits1 &= 0xffff0000
        }
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
}

}  // namespace ckernel::sfpu
