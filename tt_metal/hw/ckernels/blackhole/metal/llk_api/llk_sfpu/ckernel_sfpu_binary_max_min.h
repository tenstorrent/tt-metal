// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
    // SFPSWAP orders by sign-magnitude, so +NaN sorts above +inf and -NaN below -inf, and its two outputs are a
    // permutation of its inputs. max(-NaN, x) and min(+NaN, x) would return x; where the discarded value is NaN, it
    // replaces the result instead.
    constexpr int keep = IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0;
    constexpr int other = IS_MAX_OP ? p_sfpu::LREG0 : p_sfpu::LREG1;
    constexpr int inf = p_sfpu::LREG2;
    constexpr int scratch = p_sfpu::LREG3;

    TTI_SFPLOADI(inf, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Swap and store maximum in lreg1, minimum in lreg0
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

        // if abs(other) > inf; other is NaN
        TTI_SFPSETSGN(0, other, scratch, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPIADD(0, inf, scratch, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
        TTI_SFPMOV(0, other, keep, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        TT_SFPSTORE(keep, InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2);
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 4 cycles per input row.
    //
    // SFPSWAP orders by sign-magnitude, so +NaN sorts above +inf and -NaN below -inf: max(-NaN, x) and
    // min(+NaN, x) would return x. So after the swap, mask = -1 where the discarded value is that NaN and 0
    // elsewhere, and the result is L16 = -1.0 * mask + keep on the MAD sub-unit. A zero mask adds -0, which is
    // exact for every keep including -0; an all-ones mask is a NaN, so the result is the canonical NaN
    // 0x7fc00000. SFPMAD reads a subnormal keep as zero and returns it as a zero of the same sign.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x. L12 holds the compare bound (see the init).
    //
    // t | Load | Simple                       | MAD                  | Round | Store   |
    // - | ---- | ---------------------------- | -------------------- | ----- | ------- |
    // 0 | [a]  |                              |                      |       |         |
    // 1 | [b]  |                              |                      |       |         |
    // 2 | [c]  | swap([a], b)                 | nop                  |       |         |
    // 3 | nop  | (swap, 2nd cycle)            |                      |       |         |
    // 0 | ...  | [b] = cmp([b], L12) ? -1 : 0 |                      |       |         |
    // 1 | ...  |                              | L16 = -1.0 * b + [a] |       |         |
    // 2 | ...  | (next row's swap)            | (next row's nop)     |       |         |
    // 3 | ...  |                              |                      |       | [c] L16 |
    //
    // The swap leaves keep in [a] and the discarded value in b; it is two cycles, and the Simple and Round
    // sub-units must be idle in the second, with an SFPNOP on the MAD sub-unit beside the first
    // (SFPLOADMACRO.md, footnote ‡). a alternates between L0 and L1: it is read by the MAD one cycle after the
    // next row loads the other one. b is reloaded every row, and the nop in the 4th slot is what keeps that
    // reload (visible one cycle after the next row's [b]) after the MAD has read the mask. The store is two
    // cycles after the MAD, which takes two cycles to produce L16. c only provides the store address; the
    // value it loads is unused.

    constexpr int b = p_sfpu::LREG2;
    constexpr int c = p_sfpu::LREG3;

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0 | (a >> 2));
        TT_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1 | (b >> 2));
        TT_SFPLOADMACRO((2 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2 | (c >> 2));
        TTI_SFPNOP;
    }

    // The last store is scheduled 4 instructions after its [c] and the row's nop is the first of them; one more
    // lets it complete, as in the other SFPLOADMACRO kernels.
    TTI_SFPNOP;
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
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, offset1);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

        // Conditionally swap again to fix the cases where SFPSWAP got the result backwards
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP);
        TTI_SFPENCC(0, 0, 0, 0);

        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, offset2);
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

    load_replay_buf(0, 10, [offset0, offset1, offset2] {
        // first iteration, with a0, b0, c
        TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset0 | (a0 >> 2));
        TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset1 | (b0 >> 2));
        TTI_SFPSETCC(0, a1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset2 | (c >> 2));

        // second iteration, with a1, b1, c
        TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset0 | (a1 >> 2));
        TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset1 | (b1 >> 2));
        TTI_SFPSETCC(0, a0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset2 | (c >> 2));
    });

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
    constexpr int bound = p_sfpu::LREG12;

    // L12 = the compare bound. Max: 0xFF800001, the -NaN just below -inf, so only a -NaN is <= it. Min: +inf, so
    // only a +NaN is > it.
    if constexpr (IS_MAX_OP) {
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATB, 0xff80);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x0001);
    } else {
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);
    }
    TTI_SFPCONFIG(0, bound, 0);  // SFPCONFIG Dest 12 = LREG[12] = LREG[0]

    // InstructionTemplate[0]
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min

    // InstructionTemplate[1]: -1 where VB <= bound (SFPLE) or VB > bound (SFPGT), else 0
    if constexpr (IS_MAX_OP) {
        TTI_SFPLE(0, bound, 13, 8);  // SFPLE_MOD1_SET_VD
    } else {
        TTI_SFPGT(0, bound, 13, 8);  // SFPGT_MOD1_SET_VD
    }

    // InstructionTemplate[2]: -1.0 * b + VC
    TTI_SFPMAD(p_sfpu::LCONST_1, b, 0, 14, 1);  // SFPMAD_MOD1_NEGATE_VA

    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;
        constexpr uint mad_bits = 0x00 | 0x40 | (4 << 3) | 6;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (2 << 3) | 5;
        constexpr uint mad_bits = 0x00 | 0x00 | (0 << 3) | 2;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (4 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {1,1,1,1},
    //   UnitDelayKind: {1,1,1,1}, (WaitForElapsedInstructions=1)
    // }
    // Every delay counts issued instructions, so a stall between two SFPLOADMACROs cannot let a scheduled
    // instruction run before the load it depends on.
    TTI_SFPCONFIG(0xff0, 8, 1);

    // The configuration needs two cycles to land before the first SFPLOADMACRO.
    TTI_SFPNOP;
    TTI_SFPNOP;
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

}  // namespace sfpu
}  // namespace ckernel
