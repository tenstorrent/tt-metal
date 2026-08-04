// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
using namespace sfpi;

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit() {
    // This uses SFPLOADMACRO to achieve a throughput of 1 cycle per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t  | Load | Simple                 | MAD | Round      | Store    |
    // -- | ---- | ---------------------- | --- | ---------- | -------- |
    //  0 | [a]  |                        |     |            |          |
    //    | ...  |                        |     | [a] >>= 31 |          |
    //    | ...  | [a] L16 = cast_fp32(a) |     |            |          |
    //    | ...  |                        |     |            | [a] L16  |

    [[maybe_unused]] constexpr int offset = 0;

#ifndef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        int a = d & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset | (a >> 2));
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#else
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        // Logical-shift the fp32 bit pattern right by 31 to isolate the sign bit as 0/1,
        // then convert that integer to 0.0f / 1.0f.
        sfpi::vInt sign = sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::as<sfpi::vUInt>(in), -31));
        sfpi::dst_reg[0] = sfpi::int32_to_float(sign, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit_int32() {
    // This uses SFPLOADMACRO to achieve a throughput of 1 cycle per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t  | Load | Simple | MAD | Round             | Store    |
    // -- | ---- | ------ | --- | ----------------- | -------- |
    //  0 | [a]  |        |     |                   |          |
    //    | ...  |        |     | [a] L16 = a >> 31 |          |
    //    | ...  |        |     |                   | [a] L16  |

    [[maybe_unused]] constexpr int offset = 0;

#ifndef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr int a = p_sfpu::LREG0;
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset | (a >> 2));
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
#else
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vUInt in = sfpi::dst_reg[0];
        // Logical-shift the int32 bit pattern right by 31 to isolate the sign bit as 0/1.
        sfpi::dst_reg[0] = sfpi::shft(in, -31);
        sfpi::dst_reg++;
    }
#endif
}
inline void signbit_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPSHFT2(-31 & 0xfff, 0, 12, sfpi::SFPSHFT2_MOD1_SHFT_IMM);

    // InstructionTemplate[1]
    TTI_SFPCAST(0, 13, 0);

    // Macro 0
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (1 << 3) | 5;
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0x80 | 0x00 | (0 << 3) | 4;
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (2 << 3) | 3;

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

inline void signbit_int32_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPSHFT2(-31 & 0xfff, 0, 12, sfpi::SFPSHFT2_MOD1_SHFT_IMM);

    // Macro 0
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0x80 | 0x40 | (0 << 3) | 4;
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (1 << 3) | 3;

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

}  // namespace ckernel::sfpu
