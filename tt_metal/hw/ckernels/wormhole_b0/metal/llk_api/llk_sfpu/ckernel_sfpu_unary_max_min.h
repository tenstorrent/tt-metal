// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool IS_MAX_OP = true, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max_min(uint value) {
    // SFPSWAP orders by sign-magnitude, so max(-NaN, v) and min(+NaN, v) would return v. The swap and the store are
    // only enabled where x is not NaN, which leaves a NaN x in Dst untouched.
    constexpr int x = p_sfpu::LREG0;
    constexpr int scratch = p_sfpu::LREG1;
    constexpr int inf = p_sfpu::LREG2;

    // L12 = value, with raw instructions: through sfpi::vConstIntPrgm0 the compiler may take a constant from LREG0,
    // which it does not know the loop below overwrites.
    TT_SFPLOADI(x, sfpi::SFPLOADI_MOD0_LOWER, value & 0xffff);
    TT_SFPLOADI(x, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);
    TTI_SFPCONFIG(0, 12, 0);  // SFPCONFIG Dest 12 = LREG[12] = LREG[0]
    TTI_SFPLOADI(inf, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(x, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);

        // if abs(x) <= inf; x is not NaN
        TTI_SFPSETSGN(0, x, scratch, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPIADD(0, inf, scratch, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
        // x = max(x, constant) or min(x, constant); this will only write to x since L12 is a constant register.
        // mod1=9 means set VD=max and VC=min
        TTI_SFPSWAP(0, p_sfpu::LREG12, x, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPSTORE(x, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);  // ADDR_MOD_2 advances Dst by one row pair
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

template <bool IS_UNSIGNED = false>
sfpi_inline void load_value_param_int(uint value) {
    // if msb(value) == (IS_UNSIGNED ? 0 : 1), we need to invert for SFPSWAP to work
    sfpi::vConstIntPrgm0 = IS_UNSIGNED ^ ((int)value >= 0) ? value : ~value;
}

template <bool IS_MAX_OP, bool IS_UNSIGNED = false>
sfpi_inline void calculate_unary_max_min_int32_body(uint value) {
    sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);

    if (IS_UNSIGNED ^ ((int)value >= 0)) {
        // if msb(value) == 0, we can safely use SFPSWAP even though it expects sign-magnitude integers
        TTI_SFPSWAP(
            0,
            p_sfpu::LREG12,
            p_sfpu::LREG0,
            IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min
    } else {
        // if msb(value) == 1, we need to invert both values for SFPSWAP to work
        TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSWAP(
            0,
            p_sfpu::LREG12,
            p_sfpu::LREG0,
            IS_MAX_OP ^ IS_UNSIGNED ? sfpi::SFPSWAP_MOD1_VEC_MIN_MAX : 9);  // mod1=9 means set VD=max and VC=min
        TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    }
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max_min_int32(uint value) {
    load_value_param_int<IS_UNSIGNED>(value);

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_unary_max_min_int32_body<IS_MAX_OP, IS_UNSIGNED>(value);
        sfpi::dst_reg++;
    }
#else
    constexpr int offset = 0;

    if (IS_UNSIGNED ^ ((int)value < 0)) {
        // This uses SFPLOADMACRO to achieve a throughput of 4 cycles per input row.
        //
        // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
        //
        // t | Load | Simple                | MAD | Round | Store   |
        // - | ---- | --------------------- | --- | ----- | ------- |
        // 0 | [a]  |                       |     |       |         |
        // 1 |      | a = not(a)            |     |       |         |
        // 2 |      | swap_minmax(a, not_v) |     |       |         |
        // 3 | nop  |                       |     |       |         |
        // 0 | ...  | [a] L16 = not(a)      |     |       |         |
        // 1 | ...  |                       |     |       | [a] L16 |

        sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            constexpr int a = p_sfpu::LREG0;
            TTI_SFPLOADMACRO((1 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset | (a >> 2));
            TTI_SFPNOT(0, a, a, 0);
            TTI_SFPSWAP(
                0,
                p_sfpu::LREG12,
                a,
                IS_MAX_OP ^ IS_UNSIGNED ? sfpi::SFPSWAP_MOD1_VEC_MIN_MAX : 9);  // mod1=9 means set VD=max and VC=min
            TTI_SFPNOP;
        }
    } else {
        // This uses SFPLOADMACRO to achieve a throughput of 2 cycles per input row.
        //
        // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
        //
        // t | Load | Simple              | MAD | Round | Store |
        // - | ---- | ------------------- | --- | ----- | ----- |
        // 0 | [a]  |                     |     |       |       |
        // 1 | nop  | swap_minmax([a], v) |     |       |       |
        // 0 | ...  |                     |     |       |       |
        // 1 | ...  |                     |     |       | [a]   |

        sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
        sfpi::l_reg[sfpi::LRegs::LReg1].in_use();
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            int a = d & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
            TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset | (a >> 2));
            TTI_SFPNOP;
        }
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool IS_MAX_OP = true>
inline void unary_max_min_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void unary_max_min_int32_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPSWAP(
        0,
        p_sfpu::LREG12,
        12,
        IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min

    // InstructionTemplate[1]
    TTI_SFPNOT(0, 0, 13, 0);

    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (0 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x00 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0x00 | 0x40 | (3 << 3) | 5;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (4 << 3) | 3;

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
}  // namespace ckernel::sfpu
