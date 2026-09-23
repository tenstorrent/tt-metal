// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "lltt.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Swap and store maximum in lreg1, minimum in lreg0
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset1);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset2);
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 3 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple              | MAD | Round     | Store   |
    // - | ---- | ------------------- | --- | --------- | ------- |
    // 0 | [a]  |                     |     |           |         |
    // 1 |  b   |                     |     |           |         |
    // 2 | [c]  | swap_minmax([a], b) |     |           |         |
    // 0 | ...  |                     |     |           |         |
    // 1 | ...  |                     |     | L16 = [a] |         |
    // 2 | ...  |                     |     |           | [c] L16 |

    constexpr int b = p_sfpu::LREG2;
    constexpr int c = p_sfpu::LREG3;

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset0 | (a >> 2));
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset1);
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset2 | (c >> 2));
    }

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Swap and store maximum in lreg1, minimum in lreg0 (or reversed if unsigned)
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, offset1);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

        // Conditionally swap again to fix the cases where SFPSWAP got the result backwards
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP);
        TTI_SFPENCC(0, 0, 0, 0);

        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, offset2);
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 5 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple                | MAD | Round     | Store   |
    // - | ---- | --------------------- | --- | --------- | ------- |
    // 0 | [a0] |                       |     |           |         |
    // 1 | [b0] |                       |     |           |         |
    // 2 |      | setcc  a1  (<0 or ≥0) |     |           |         |
    // 3 |      | encc                  |     |           |         |
    // 4 | [c]  | swap_minmax([a0], b0) |     |           |         |
    // 0 | ...  |                       |     |           |         |
    // 1 | ...  | setcc [b0] (<0 or ≥0) |     | L16 = [a] |         |
    // 2 | ...  | setcc  a0  (<0 or ≥0) |     |           |         |
    // 3 | ...  | encc                  |     | L16 = [b] |         |
    // 4 | ...  |                       |     |           | [c] L16 |

    constexpr int a0 = p_sfpu::LREG0;
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int a1 = p_sfpu::LREG2;
    constexpr int b1 = p_sfpu::LREG3;
    constexpr int c = p_sfpu::LREG7;

    lltt::record<lltt::NoExec>(0, 10);

    // first iteration, with a0, b0, c
    TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a0 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b0 >> 2));
    TTI_SFPSETCC(0, a1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset2 | (c >> 2));

    // second iteration, with a1, b1, c
    TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a1 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b1 >> 2));
    TTI_SFPSETCC(0, a0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset2 | (c >> 2));

#pragma GCC unroll 4
    for (int i = 0; i < ITERATIONS / 2; ++i) {
        lltt::replay(0, 10);
    }

    if constexpr (ITERATIONS & 1) {
        lltt::replay(0, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(5 + 2, 2);
    } else {
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(2, 2);
    }

    TTI_SFPNOP;
#endif
}

template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b = p_sfpu::LREG2;

    // InstructionTemplate[0]
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min

    // InstructionTemplate[1]
    TTI_SFPSHFT2(0, 0, 13, 6);  // SFPSHFT2_MOD1_SHFT_IMM

    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 5;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {1,1},
    //   UnitDelayKind: {1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x330, 8, 1);
#endif
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void binary_max_min_int32_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int b1 = p_sfpu::LREG3;

    // InstructionTemplate[0]
    TTI_SFPSWAP(
        0, b0, 12, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min

    // InstructionTemplate[1]
    TTI_SFPSWAP(
        0, b1, 13, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min

    // InstructionTemplate[2]
    TTI_SFPSETCC(0, 0, 14, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);

    // InstructionTemplate[3]
    TTI_SFPSHFT2(0, 0, 15, 6);  // SFPSHFT2_MOD1_SHFT_IMM

    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 5;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2:
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (4 << 3) | 6;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (6 << 3) | 7;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Macro 3:
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (4 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {1,1,1,1},
    //   UnitDelayKind: {1,1,1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0xff0, 8, 1);
#endif
}

// ---------------------------------------------------------------------------------------------------
// Approach A dispatch struct (prototype).
// BinaryMaxMin<IS_MAX_OP, FORMAT, DST_SYNC, DST_ACCUM>::calculate(in0, in1, out, vector_mode)
// BinaryMaxMin<...>::init()
// FORMAT: Float16_b/Float32 -> float kernel, Int32 -> signed int32 kernel, UInt32 -> unsigned int32 kernel.
// ---------------------------------------------------------------------------------------------------
template <bool IS_MAX_OP, DataFormat FORMAT, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct BinaryMaxMin
    : SfpuBinaryOp<BinaryMaxMin<IS_MAX_OP, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static constexpr bool is_int = FORMAT == DataFormat::Int32 || FORMAT == DataFormat::UInt32;
    static constexpr bool is_unsigned = FORMAT == DataFormat::UInt32;

    static void kernel(uint32_t dst_index_in0, uint32_t dst_index_in1, uint32_t dst_index_out) {
        if constexpr (is_int) {
            calculate_binary_max_min_int32<IS_MAX_OP, is_unsigned, ITERATIONS>(
                dst_index_in0, dst_index_in1, dst_index_out);
        } else {
            calculate_binary_max_min<IS_MAX_OP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
        }
    }

    static void init_kernel() {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
        if constexpr (is_int) {
            binary_max_min_int32_init<IS_MAX_OP, is_unsigned>();
        } else {
            binary_max_min_init<IS_MAX_OP>();
        }
    }
};
}  // namespace sfpu
}  // namespace ckernel
