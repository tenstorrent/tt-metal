// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
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

    // A float condition is true unless it is +0.0 or -0.0, so its sign bit is cleared (SFPSETSGN) before the
    // bitwise zero test. A bf16 condition is loaded with FP16B to put its sign in bit 31; both FP16B and INT32 move
    // bits without conversion, so NaN and subnormal conditions stay non-zero. Integer conditions keep the bitwise
    // test on the raw value.
    constexpr bool cond_is_float = data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b;
    constexpr InstrModLoadStore cond_mod = data_format == DataFormat::Float16_b ? InstrModLoadStore::FP16B : mod0;
    constexpr std::uint32_t SFPSETSGN_MOD1_ARG_IMM = 1;

    int offset3 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
    // Three SFPLOADs, the condition test (SFPSETSGN for floats, SFPSETCC), SFPENCC, SFPSTORE.
    constexpr std::uint32_t replay_len = cond_is_float ? 7 : 6;

    TTI_REPLAY(0, replay_len, 0, 1);
    TT_SFPLOAD(p_sfpu::LREG0, cond_mod, ADDR_MOD_3, offset0);
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, offset1);
    if constexpr (cond_is_float) {
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPSETSGN_MOD1_ARG_IMM);
    }
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, offset2);
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
    TT_SFPSTORE(p_sfpu::LREG1, mod0, ADDR_MOD_2, offset3);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_REPLAY(0 /*start*/, replay_len, 0, 0);
    }
#else
    // Every macro below loads into the register named in the SFPLOADMACRO. Macros 0 and 1 run InstructionTemplate[0],
    // SFPSETCC on L1, so the condition always sits in L1.
    if constexpr (cond_is_float) {
        // Macro 3 loads the condition and clears its sign; macro 1 tests it while loading a dummy, so the true
        // value (macro 2) lands on the cycle the flags change and is not masked; the false value is then masked.
        // 5 cycles per input row of 32 values, for both in-place and separate outputs:

        // Load Unit               | Simple Unit                       | Store Unit
        // SFPLOAD L1=Dst[offset0] |                                   |
        // SFPLOAD L2=Dst[offset1] | SFPSETSGN L1=+|L1|                |
        // SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L1 EQ 0)    |
        // SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)        |
        // -                       |                                   | SFPSTORE Dst[offset3]=L0

        TTI_REPLAY(0, 5, 0, 1);
        TT_SFPLOADMACRO((3 << 2) | p_sfpu::LREG1, cond_mod, ADDR_MOD_3, offset0);
        TT_SFPLOADMACRO((1 << 2) | p_sfpu::LREG2, mod0, ADDR_MOD_3, offset1);
        TT_SFPLOADMACRO((2 << 2) | p_sfpu::LREG0, mod0, ADDR_MOD_3, offset1);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, offset2);
        TT_SFPSTORE(p_sfpu::LREG0, mod0, ADDR_MOD_2, offset3);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_REPLAY(0, 5, 0, 0);
        }
    } else if (dst_index_out == dst_index_in0) {
        // Macros 0 and 2 schedule the following, which achieves 3 cycles per input row of 32 values:

        // Load Unit               | Simple Unit                       | Store Unit
        // SFPLOAD L1=Dst[offset0] |                                   |
        // SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L1 EQ 0)    |
        // SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)        |
        // (next SFPLOAD L1)       |                                   | SFPSTORE Dst[offset0]=L0

        TTI_REPLAY(0, 3, 0, 1);
        TT_SFPLOADMACRO((0 << 2) | p_sfpu::LREG1, mod0, ADDR_MOD_3, offset0);
        TT_SFPLOADMACRO((2 << 2) | p_sfpu::LREG0, mod0, ADDR_MOD_3, offset1);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_2, offset2);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_REPLAY(0, 3, 0, 0);
        }
    } else {
        // Macros 1 and 2 schedule the following, which achieves 4 cycles per input row of 32 values:

        // Load Unit               | Simple Unit                       | Store Unit
        // SFPLOAD L1=Dst[offset0] |                                   |
        // SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L1 EQ 0)    |
        // SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)        |
        // -                       |                                   | SFPSTORE Dst[offset3]=L0

        TTI_REPLAY(0, 4, 0, 1);
        TT_SFPLOADMACRO((1 << 2) | p_sfpu::LREG1, mod0, ADDR_MOD_3, offset0);
        TT_SFPLOADMACRO((2 << 2) | p_sfpu::LREG0, mod0, ADDR_MOD_3, offset1);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, offset2);
        TT_SFPSTORE(p_sfpu::LREG0, mod0, ADDR_MOD_2, offset3);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_REPLAY(0, 4, 0, 0);
        }
    }
#endif
}

template <bool APPROXIMATION_MODE>
inline void _init_where_() {
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]: zero test of L1. Macros 0 and 1 set sequence bit 0x80 so VC stays L1 rather than the
    // macro's register.
    TTI_SFPSETCC(0, p_sfpu::LREG1, 12, sfpi::SFPSETCC_MOD1_LREG_EQ0);

    // InstructionTemplate[1]
    TTI_SFPENCC(0, 0, 13, 0);

    // InstructionTemplate[2]: clear the sign bit of the macro's register (SFPSETSGN with an immediate 0 sign).
    TTI_SFPSETSGN(0, 0, 14, 1);

    constexpr std::uint32_t mad_bits = 0;
    constexpr std::uint32_t round_bits = 0;

    // Macro 0: InstructionTemplate[0], then store L0 to the loaded address two cycles later. Store bit 0x80 keeps
    // the store's VD at L0 rather than the macro's register.
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | 4;
        constexpr std::uint32_t store_bits = 0x80 | 0x00 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1: InstructionTemplate[0].
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | 4;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }

    // Macro 2: InstructionTemplate[1].
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 5;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);
    }

    // Macro 3: InstructionTemplate[2].
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 6;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 3, 1);
    }

    // Misc: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for all macros.
    TTI_SFPCONFIG(0x770, 8, 1);
#endif
}

}  // namespace ckernel::sfpu
