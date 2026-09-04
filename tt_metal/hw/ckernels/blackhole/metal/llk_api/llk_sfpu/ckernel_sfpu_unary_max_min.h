// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

sfpi_inline void load_value_param_float(uint value) { sfpi::vConstIntPrgm0 = value; }

template <bool IS_MAX_OP>
sfpi_inline void calculate_unary_max_min_float_body() {
    sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

    if constexpr (IS_MAX_OP) {
        // L0 = max(L0, constant); this will only write to L0 since L12 is a constant register.
        TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, 9);  // mod1=9 means set VD=max and VC=min
    } else {
        // L0 = min(L0, constant); this will only write to L0 since L12 is a constant register.
        TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    }
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
}

template <bool IS_MAX_OP = true, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max_min(uint value) {
    // This uses SFPLOADMACRO to achieve a throughput of 2 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t  | Load | Simple              | MAD | Round | Store |
    // -- | ---- | ------------------- | --- | ----- | ----- |
    //  0 | [a]  |                     |     |       |       |
    //  1 | nop  | swap_minmax([a], v) |     |       |       |
    //  0 | ...  |                     |     |       |       |
    //  1 | ...  |                     |     |       | [a]   |

    load_value_param_float(value);
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_unary_max_min_float_body<IS_MAX_OP>();
        sfpi::dst_reg++;
    }
#else
    constexpr int offset = 0;

    sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
    sfpi::l_reg[sfpi::LRegs::LReg1].in_use();
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        int a = d & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset | (a >> 2));
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool IS_UNSIGNED = false>
sfpi_inline void load_value_param_int(uint value) {
    // if msb(value) == (IS_UNSIGNED ? 0 : 1), we need to invert for SFPSWAP to work
    sfpi::vConstIntPrgm0 = IS_UNSIGNED ^ ((int)value >= 0) ? value : ~value;
}

template <bool IS_MAX_OP, bool IS_UNSIGNED = false>
sfpi_inline void calculate_unary_max_min_int32_body(uint value) {
    sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);

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
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
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
            TTI_SFPLOADMACRO((1 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset | (a >> 2));
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
            TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset | (a >> 2));
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
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPSWAP(
        0,
        p_sfpu::LREG12,
        12,
        IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min

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

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {1},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x110, 8, 1);
#endif
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
// UnaryMaxMin<IS_MAX_OP, FORMAT, APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>: unary_max_tile / unary_min_tile
// (Float16_b), unary_{max,min}_int32_tile (Int32), unary_{max,min}_uint32_tile (UInt32) and their *_init
// entry points in compute_kernel_api.h. FORMAT selects the float or int32/uint32 kernel + init.
template <
    bool IS_MAX_OP,
    DataFormat FORMAT,
    bool APPROXIMATION_MODE,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = 8>
struct UnaryMaxMin : SfpuUnaryOp<
                         UnaryMaxMin<IS_MAX_OP, FORMAT, APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>,
                         DST_SYNC,
                         DST_ACCUM> {
    static_assert(
        FORMAT == DataFormat::Float16_b || FORMAT == DataFormat::Int32 || FORMAT == DataFormat::UInt32,
        "UnaryMaxMin supports Float16_b, Int32 and UInt32");

    static constexpr bool is_unsigned = (FORMAT == DataFormat::UInt32);
    static constexpr bool is_int = (FORMAT == DataFormat::Int32) || is_unsigned;

    static void kernel(uint32_t value) {
        if constexpr (is_int) {
            calculate_unary_max_min_int32<IS_MAX_OP, is_unsigned, APPROXIMATION_MODE, ITERATIONS>(value);
        } else {
            calculate_unary_max_min<IS_MAX_OP, APPROXIMATION_MODE, ITERATIONS>(value);
        }
    }

    static void init_kernel() {
        if constexpr (is_int) {
            unary_max_min_int32_init<IS_MAX_OP, is_unsigned>();
        } else {
            unary_max_min_init<IS_MAX_OP>();
        }
    }
};

}  // namespace ckernel::sfpu
