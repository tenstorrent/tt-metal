// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// probability should be between 0 - INT_MAX (signed)
// scale should be binary representation of a float32
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_dropout(uint probability, uint scale) {
    // SFPU microcode

    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);
    // Unrolled: the RISC then issues the eight-instruction row bodies back to back instead of paying the
    // loop iteration per row that idled the SFPU about two cycles per row.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        ////////////////////////
        // Scale samples
        // dst_reg[0] = dst_reg[0] * sFloat16b(scale);
        ///////////////////////
        // DEST is addressed through ADDR_MOD_7, the slot the SFPU inits program to a zero DEST step on
        // Blackhole. Slot 3 is the Wormhole SFPU slot; here it belongs to the FPU programs, and the eltwise
        // binary init sets it to a DEST step of 8 rows, so a body that used it read and wrote the wrong rows
        // when dropout ran after add_tiles in one DEST section.
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        ////////////////////////
        // Instruction SFPMOV generates a uint32_t pseudorandom number
        // when instr_mod1 = 8 and lreg_c =  9.
        // Arguments: (imm12_math, lreg_c, lreg_dest, instr_mod1)
        // Unset sign-bit for easy comparison with probability
        ////////////////////////
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        ////////////////////////
        // Drop samples
        // v_if (rand < probability)
        //   dst_reg[0] = 0.0f;
        ///////////////////////
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void dropout_init(const uint seed) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    init_prng_seed(seed);
}

}  // namespace sfpu
}  // namespace ckernel
