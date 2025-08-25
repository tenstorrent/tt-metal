// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

template <typename V, typename T>
inline void calculate_logical_not_unary() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        V v = sfpi::dst_reg[0];
        v_if(v == 0) { sfpi::dst_reg[0] = T(1); }
        v_else { sfpi::dst_reg[0] = T(0); }
        v_endif;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logical_not_unary_uint16() {
    for (int d = 0; d < ITERATIONS; d++) {
        // full tile size
        constexpr int tile_size = 64;
        // location of conditional value tile
        constexpr int cond_val_idx = 0 * tile_size;  // tile 0
        // location of output tile
        constexpr int output_idx = 0 * tile_size;  // tile 0
        // load in conditional uint16 value
        TTI_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, cond_val_idx);
        // initially put 0 into output
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        // if (REG0 == 0)
        TTI_SFPSETCC(0, 0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        // load in (int) 1
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0x0001);
        // end_if
        TTI_SFPENCC(0, 0, 0, 0);
        // store result
        TTI_SFPSTORE(p_sfpu::LREG1, LO16, ADDR_MOD_3, output_idx);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
