// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Relational ops use int32 subtract (whose result in also in the int32 range) + sign check.
// In order to avoid overflow for inputs of opposite signs, the output is determined directly from a sign check.
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64 rows
        constexpr uint dst_tile_size = 64;
        // operand A
        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        // operand B
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);

        if constexpr (RELATIONAL_OP == SfpuType::lt) {
            // Extract sign bits of A and B
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);

            // LREG_3 -> 0 for inputs of same sign, 1 for inputs of different signs
            TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);

            // if (LREG_3 == 0) -> use int32 subtract + extract sign
            TTI_SFPSETCC(0, p_sfpu::LREG3, 0 /*unused*/, SFPSETCC_MOD1_LREG_EQ0);
            // (A - B) -> Use 6 or LO16 as imod to convert operand B to 2's complement
            TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
            // else -> load 0 for inputs of opposite signs
            TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
            TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x00);
            // Load 1 if input A is negative
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPENCC(0, 0, 0, 0);

            // LREG_1 -> dest
            TT_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);
        } else if constexpr (RELATIONAL_OP == SfpuType::gt) {
            // Extract sign bits of A and B
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);

            // LREG_3 -> 0 for inputs of same sign, 1 for inputs of different signs
            TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);

            // if (LREG_3 == 0) -> use int32 subtract + extract sign
            TTI_SFPSETCC(0, p_sfpu::LREG3, 0, SFPSETCC_MOD1_LREG_EQ0);
            // (B - A) -> Use 6 or LO16 as imod to convert operand B to 2's complement
            TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
            // else -> load 0 for inputs of opposite signs
            TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
            TTI_SFPLOADI(p_sfpu::LREG0, SFPLOADI_MOD0_USHORT, 0x00);
            // Load 1 if input A is non-negative
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPLOADI(p_sfpu::LREG0, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPENCC(0, 0, 0, 0);

            // LREG_0 -> dest
            TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);
        }

        sfpi::dst_reg++;
    }
}

}  //  namespace ckernel::sfpu
