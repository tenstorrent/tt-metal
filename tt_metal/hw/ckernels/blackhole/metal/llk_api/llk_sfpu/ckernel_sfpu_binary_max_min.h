// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    /*
    #pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            // size of each tile in Dest is 64 rows
            constexpr uint dst_tile_size = 64;

            TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_in0 * dst_tile_size);  // a
            TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_in1 * dst_tile_size);  // b

            // Swap and store maximum in lreg1, minimum in lreg0
            TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

            if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32) {
                // The values are two's complement signed integers, but SFPSWAP
                // treats them as sign-magnitude.  The result is still correct
                // unless both values are negative, in which case the result simply
                // needs to be inverted via unconditional swap.

                TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
                TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);

                // If a < 0 and b < 0, then invert the result.
                TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP);

                TTI_SFPENCC(0, 0, 0, 0);
            }

            if constexpr (IS_MAX_OP) {
                TT_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_out * dst_tile_size);
            } else {
                TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_out * dst_tile_size);
            }
            dst_reg++;
        }
    */
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr int a = 0;
        constexpr int b = 1;
        constexpr int c = 2;
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 10 /* MOD0_FMT_INT32_ALL */, ADDR_MOD_7, offset0 | (a >> 2));
        TT_SFPLOADMACRO((1 << 2) | (b & 3), 10 /* MOD0_FMT_INT32_ALL */, ADDR_MOD_7, offset1 | (b >> 2));
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPNOP;
        TT_SFPLOADMACRO((2 << 2) | (c & 3), 10 /* MOD0_FMT_INT32_ALL */, ADDR_MOD_6, offset2 | (c >> 2));
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void binary_max_min_int32_init() {
    // InstructionTemplate[0]
    TTI_SFPSETCC(0, 0, 12, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);

    // InstructionTemplate[1]
    TTI_SFPENCC(0, 0, 13, 0);

    // InstructionTemplate[2]
    TTI_SFPSHFT2(0, 0, 14, 6);  // SFPSHFT2_MOD1_SHFT_IMM

    // Macro 0
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (3 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 6;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (3 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (4 << 3) | 6;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2:
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (1 << 3) | 5;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Misc: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for all macros.
    TTI_SFPCONFIG(0x770, 8, 1);
}

}  // namespace sfpu
}  // namespace ckernel
