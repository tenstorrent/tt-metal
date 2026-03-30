// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
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

    constexpr int offset = 0;

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
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset);
        TTI_SFPSHFT(-31 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);  // SFPSHFT_MOD1_ARG_IMM
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset);
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

    constexpr int offset = 0;

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
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, offset);
        TTI_SFPSHFT(-31 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);  // SFPSHFT_MOD1_ARG_IMM
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, offset);
    }
#endif
}

inline void signbit_init() {
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPSHFT2(-31 & 0xfff, 0, 12, sfpi::SFPSHFT2_MOD1_SHFT_IMM);

    // InstructionTemplate[1]
    TTI_SFPCAST(0, 13, 0);

    // Macro 0
    {
        constexpr uint simple_bits = 0x00 | 0x40 | (1 << 3) | 5;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x00 | (0 << 3) | 4;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;

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
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPSHFT2(-31 & 0xfff, 0, 12, sfpi::SFPSHFT2_MOD1_SHFT_IMM);

    // Macro 0
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (0 << 3) | 4;
        constexpr uint store_bits = 0x00 | 0x40 | (1 << 3) | 3;

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
