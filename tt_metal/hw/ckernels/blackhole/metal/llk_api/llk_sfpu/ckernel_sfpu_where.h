// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_in2,
    const std::uint32_t dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b ||
            data_format == DataFormat::Int32 || data_format == DataFormat::UInt32,
        "Unsupported data format for _calculate_where_(). Only Float32, Int32, UInt32, and Float16_b are allowed.");

    int offset0 = (dst_index_in0 * 32) << 1;
    int offset1 = (dst_index_in1 * 32) << 1;
    int offset2 = (dst_index_in2 * 32) << 1;

    constexpr InstrModLoadStore mod0 =
        data_format == DataFormat::Float16_b ? InstrModLoadStore::UINT16 : InstrModLoadStore::INT32;

    // Only Float32 is unpacked directly to Dest (unpack_to_dest=true), preserving IEEE-754 -0.0 (0x80000000).
    // Float16_b unpacks via srcA (MOVA2D) which flushes -0.0 before Dest, and Int32/UInt32 have no signed zero.
    constexpr bool is_fp32 = (data_format == DataFormat::Float32);

#ifdef DISABLE_SFPLOADMACRO
    int offset3 = (dst_index_out * 32) << 1;

    // Three SFPLOADs, optional SFPSHFT (for Float32), SFPSETCC, SFPENCC, SFPSTORE.
    constexpr std::uint32_t replay_len = is_fp32 ? 7 : 6;

    // Same bracketed record as the SFPLOADMACRO paths below: load_replay_buf disables instruction
    // gathering around the record window when the kernel is built with gathering on.
    load_replay_buf(0 /*start*/, replay_len, [offset0, offset1, offset2, offset3] {
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset1);
        if constexpr (is_fp32) {
            // Drop the IEEE-754 sign bit so that -0.0 (0x80000000) evaluates to 0 (false)
            TTI_SFPSHFT(1, p_sfpu::LREG0, p_sfpu::LREG0, 1);
        }
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset2);
        TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
        TT_SFPSTORE(p_sfpu::LREG1, mod0, ADDR_MOD_6, offset3);
    });

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        lltt::replay(0 /*start*/, replay_len);
    }
#else
    if (dst_index_out == dst_index_in0) {
        // Non-Float32 uses macros 0 and 2 (3 cycles per row); Float32 uses macros 3 and 2 (4 cycles per row)
        // to insert SFPSHFT clearing the sign bit of -0.0 before SFPSETCC:
        //
        // Load Unit               | Simple Unit                       | Store Unit
        // SFPLOAD L0=Dst[offset0] |                                   |
        // - (Float32 only)        | SFPSHFT (clears sign bit: ±0->0)  |
        // SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L0 EQ 0)    |
        // SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)        |
        // (next SFPLOAD L0)       |                                   | SFPSTORE Dst[offset0]=L0

        constexpr std::uint32_t replay_len = is_fp32 ? 4 : 3;

        load_replay_buf(0, replay_len, [offset0, offset1, offset2] {
            if constexpr (is_fp32) {
                TT_SFPLOADMACRO((3 << 2), mod0, ADDR_MOD_7, offset0);
                TTI_SFPSHFT(1, p_sfpu::LREG0, p_sfpu::LREG0, 1);
            } else {
                TT_SFPLOADMACRO((0 << 2), mod0, ADDR_MOD_7, offset0);
            }
            TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1);
            TT_SFPLOAD(0, mod0, ADDR_MOD_6, offset2);
        });

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            lltt::replay(0, replay_len);
        }
    } else {
        // Non-Float32 uses macros 1 and 2 (4 cycles per row); Float32 adds SFPSHFT (5 cycles per row):
        //
        // Load Unit               | Simple Unit                       | Store Unit
        // SFPLOAD L0=Dst[offset0] |                                   |
        // - (Float32 only)        | SFPSHFT (clears sign bit: ±0->0)  |
        // SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L0 EQ 0)    |
        // SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)        |
        // -                       |                                   | SFPSTORE Dst[offset3]=L0
        // (next SFPLOAD L0)       |                                   |

        int offset3 = (dst_index_out * 32) << 1;
        constexpr std::uint32_t replay_len = is_fp32 ? 5 : 4;

        if constexpr (is_fp32) {
            TTI_SFPCONFIG((1 << 3) | 4, 4 + 1, 1);
            TTI_SFPNOP;
        }

        load_replay_buf(0, replay_len, [offset0, offset1, offset2, offset3] {
            TT_SFPLOADMACRO((1 << 2), mod0, ADDR_MOD_7, offset0);
            if constexpr (is_fp32) {
                TTI_SFPSHFT(1, p_sfpu::LREG0, p_sfpu::LREG0, 1);
            }
            TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1);
            TT_SFPLOAD(0, mod0, ADDR_MOD_7, offset2);
            TT_SFPSTORE(0, mod0, ADDR_MOD_6, offset3);
        });

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            lltt::replay(0, replay_len);
        }

        if constexpr (is_fp32) {
            TTI_SFPCONFIG((0 << 3) | 4, 4 + 1, 1);
        }
    }
#endif
}

template <bool APPROXIMATION_MODE>
inline void _init_where_() {
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPSETCC(0, 0, 12, 6);  // SFPSETCC_MOD1_LREG_EQ0

    // InstructionTemplate[1]
    TTI_SFPENCC(0, 0, 13, 0);

    // Macro 0: special case handling for where(a, b, c, a), i.e. write the output to the first input (non-Float32).
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4;
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1: otherwise, handle where(a, b, c, d).
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4;
        constexpr std::uint32_t mad_bits = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }

    // Macro 2:
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 5;
        constexpr std::uint32_t mad_bits = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);
    }

    // Macro 3: Float32 handling for where(a, b, c, a) with +1 delay for the sign-bit clearing SFPSHFT step.
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (1 << 3) | 4;
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for all 4 macros.
    TTI_SFPCONFIG(0xFF0, 8, 1);
#endif
}

}  // namespace ckernel::sfpu
