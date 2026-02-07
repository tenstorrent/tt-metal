// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Comparison ops use int32 subtract (whose result is also in the int32 range) + sign check.
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

        // Extract sign bits of A and B
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        // LREG_3 -> 0 for inputs of same sign, 1 for inputs of different signs
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);

        if constexpr (RELATIONAL_OP == SfpuType::lt) {
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

            // LREG_1 -> dest
            TT_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);

        } else if constexpr (RELATIONAL_OP == SfpuType::gt) {
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

            // LREG_0 -> dest
            TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);

        } else if constexpr (RELATIONAL_OP == SfpuType::ge) {
            // Implements GE by using LT logic and then inverting the result
            TTI_SFPSETCC(0, p_sfpu::LREG3, 0 /*unused*/, SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
            TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
            TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x00);
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPENCC(0, 0, 0, 0);

            // XOR with 1 to invert the result
            TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPXOR(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);

            // LREG_1 -> dest
            TT_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);

        } else if constexpr (RELATIONAL_OP == SfpuType::le) {
            // Implements LE by using GT logic and then inverting the result
            TTI_SFPSETCC(0, p_sfpu::LREG3, 0, SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
            TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
            TTI_SFPLOADI(p_sfpu::LREG0, SFPLOADI_MOD0_USHORT, 0x00);
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPLOADI(p_sfpu::LREG0, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPENCC(0, 0, 0, 0);

            // XOR with 1 to invert the result
            TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPXOR(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);

            // LREG_0 -> dest
            TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);
        }

        sfpi::dst_reg++;
    }
}

// Float32 binary comparison
// TODO: Add support for ne, gt, lt, ge, le operations
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_fp32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(RELATIONAL_OP == SfpuType::eq, "Supported operation types: eq ");
    constexpr uint dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f;

        if constexpr (RELATIONAL_OP == SfpuType::eq) {
            sfpi::vInt in0_bits = sfpi::reinterpret<sfpi::vInt>(in0);
            sfpi::vInt in1_bits = sfpi::reinterpret<sfpi::vInt>(in1);

            // Standard float comparison (handles normal values and NaN correctly)
            v_if(in0 == in1) { result = 1.0f; }
            // Special handling for infinity
            v_elseif((in0_bits == in1_bits) && ((in0_bits & 0x7FFFFFFF) == 0x7F800000)) { result = 1.0f; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

}  //  namespace ckernel::sfpu
