// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE, int ITERATIONS>
inline void calculate_logical_not() {
    static_assert(
        INSTRUCTION_MODE == InstrModLoadStore::DEFAULT || INSTRUCTION_MODE == InstrModLoadStore::LO16 ||
            INSTRUCTION_MODE == InstrModLoadStore::INT32,
        "INSTRUCTION_MODE must be one of: DEFAULT, LO16, INT32.");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr int tile_size = 64;
        // load in conditional uint16 value
        TTI_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, 0);
        // initially put 0 into output
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        // if (REG0 == 0)
        TTI_SFPSETCC(0, 0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        // load in 1
        if constexpr (INSTRUCTION_MODE == InstrModLoadStore::DEFAULT) {
            TTI_SFPMOV(0, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
        } else {
            TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0x0001);
        }
        // TTI_SFPENCC(IMM12_MATH, LREG_C, LREG_DEST, INSTR_MOD1);
        // IMM12_MATH: optional immediate value for math operations
        // LREG_C: unused
        // LREG_DEST: unused
        // INSTR_MOD1: 0 => condition code enable reg is not modified.
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
