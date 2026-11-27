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

    // A float condition is true unless it is +0.0 or -0.0, so its zero test must ignore the sign bit: SFPLZ with
    // NOSGN masks bit 31. A bf16 condition is loaded with HI16 so its sign lands in bit 31. Integer conditions keep
    // the bitwise SFPSETCC test.
    constexpr bool cond_is_float = data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b;
    constexpr InstrModLoadStore cond_mod =
        data_format == DataFormat::Float16_b ? InstrModLoadStore::HI16 : InstrModLoadStore::INT32;

#ifdef DISABLE_SFPLOADMACRO
    int offset3 = (dst_index_out * 32) << 1;

    TTI_REPLAY(0, 6, 0, 1);
    TT_SFPLOAD(p_sfpu::LREG0, cond_mod, ADDR_MOD_3, offset0);
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, offset1);
    if constexpr (cond_is_float) {
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPLZ_MOD1_NOSGN_CC_EQ0);
    } else {
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    }
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, offset2);
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
    TT_SFPSTORE(p_sfpu::LREG1, mod0, ADDR_MOD_2, offset3);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_REPLAY(0, 6, 0, 0);
    }
#else
    // The condition is loaded into L1: SFPLZ writes its VD, which a macro sets to the register it loaded, on the
    // same cycle that L0 receives the true value.
    constexpr std::uint32_t cond_macro =
        data_format == DataFormat::Float16_b ? 3 : (data_format == DataFormat::Float32 ? 1 : 0);

    if (dst_index_out == dst_index_in0) {
        // Macro 0, 1 or 3 and macro 2 schedule the following, which achieves 3 cycles per input row of 32 values:

        // Load Unit               | Simple Unit                       | Store Unit
        // SFPLOAD L1=Dst[offset0] |                                   |
        // SFPLOAD L0=Dst[offset1] | SFPSETCC / SFPLZ on L1 (EQ 0)     |
        // SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)        |
        // (next SFPLOAD L1)       |                                   | SFPSTORE Dst[offset0]=L0

        TTI_REPLAY(0, 3, 0, 1);
        TT_SFPLOADMACRO((cond_macro << 2) | p_sfpu::LREG1, cond_mod, ADDR_MOD_3, offset0);
        TT_SFPLOADMACRO((2 << 2) | p_sfpu::LREG0, mod0, ADDR_MOD_3, offset1);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_2, offset2);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_REPLAY(0, 3, 0, 0);
        }
    } else {
        // The macros that test the condition also store to its address, so this path issues every instruction
        // directly. The true value is loaded before the test: SFPLOAD only writes enabled lanes. 6 cycles per input
        // row of 32 values.
        int offset3 = (dst_index_out * 32) << 1;

        TTI_REPLAY(0, 6, 0, 1);
        TT_SFPLOAD(p_sfpu::LREG1, cond_mod, ADDR_MOD_3, offset0);
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, offset1);
        if constexpr (cond_is_float) {
            TTI_SFPLZ(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPLZ_MOD1_NOSGN_CC_EQ0);
        } else {
            TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        }
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, offset2);
        TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
        TT_SFPSTORE(p_sfpu::LREG0, mod0, ADDR_MOD_2, offset3);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_REPLAY(0, 6, 0, 0);
        }
    }
#endif
}

template <bool APPROXIMATION_MODE>
inline void _init_where_() {
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]: integer condition, bitwise zero test.
    TTI_SFPSETCC(0, 0, 12, sfpi::SFPSETCC_MOD1_LREG_EQ0);

    // InstructionTemplate[1]
    TTI_SFPENCC(0, 0, 13, 0);

    // InstructionTemplate[2]: float condition, zero test that ignores the sign bit so +-0.0 are both false.
    TTI_SFPLZ(0, 0, 14, sfpi::SFPLZ_MOD1_NOSGN_CC_EQ0);

    // Macros 0 (integer), 1 (fp32) and 3 (bf16) test the condition and store L0 back to the condition's address two
    // cycles later. Store bit 0x80 keeps the store's VD at L0 rather than the macro's VD (L1).
    constexpr std::uint32_t round_bits = 0;
    constexpr std::uint32_t mad_bits = 0;
    constexpr std::uint32_t store_bits = 0x80 | 0x00 | (2 << 3) | 3;
    constexpr std::uint32_t int_cond_bits = 0x00 | 0x00 | (0 << 3) | 4;
    constexpr std::uint32_t float_cond_bits = 0x00 | 0x00 | (0 << 3) | 6;

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | int_cond_bits);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
    TTI_SFPCONFIG(0, 4 + 0, 0);

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | float_cond_bits);
    TTI_SFPCONFIG(0, 4 + 1, 0);
    TTI_SFPCONFIG(0, 4 + 3, 0);

    // Macro 2: true-value load + InstructionTemplate[1].
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 5;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);
    }

    // Misc: StoreMod0=UINT16, UsesLoadMod0ForStore for macros 0-2, WaitForElapsedInstructions. Macros 0 and 1 load
    // the condition with INT32 and store with it. Macro 3 loads a bf16 condition with HI16 and must store with
    // UINT16.
    TTI_SFPCONFIG(0x770 | static_cast<std::uint32_t>(InstrModLoadStore::UINT16), 8, 1);
#endif
}

}  // namespace ckernel::sfpu
