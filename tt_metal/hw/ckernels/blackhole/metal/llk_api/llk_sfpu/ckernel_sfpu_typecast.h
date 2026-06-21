// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Mask that keeps the low 16 bits (the UInt16 value) of a 32-bit dest word and clears the garbage high bits.
constexpr std::uint16_t UINT16_LOW_MASK = 0xFFFF;

// SFPSTORE mode that swaps the high and low 16 bits before writing, so a value computed in the low 16 bits
// lands in the high 16 bits where the packer reads UInt16 out of a 32-bit dest word.
constexpr std::uint32_t SFPSTORE_MODE_SWAP_HI_LO16 = 9;

// SFPGT mod1 selector that sets the destination to all-ones (-1) when the comparison is true.
constexpr std::uint32_t SFPGT_MOD1_SET_ALL_ONES = 8;

template <bool APPROXIMATION_MODE, int ITERATIONS, bool DST_ACCUM_MODE>
inline void calculate_typecast_fp32_to_uint16() {
    // TODO: Attempt to use LOADMACRO #46751
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 9);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
        if (DST_ACCUM_MODE) {
            TTI_SFPSTORE(p_sfpu::LREG0, SFPSTORE_MODE_SWAP_HI_LO16, ADDR_MOD_6, 0);
        } else {
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_6, 0);
        }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp16b() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 1 cycle per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple        | MAD | Round            | Store   |
    // - | ---- | ------------- | --- | ---------------- | ------- |
    // 0 | [v]  |               |     |                  |         |
    // 0 | ...  | [v] = cast(v) |     |                  |         |
    // 0 | ...  |               |     | [v] L16 = rnd(v) |         |
    // 0 | ...  |               |     |                  | [v] L16 |

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        int v = d & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp16b() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);                   // lreg[1] = iabs(lreg[0])
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, 0);                     // lreg[2] = cast(lreg[1])
        TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);                // lreg[0] = sign(lreg[0]) | exp_man(lreg[2])
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // cc = lreg[1] < 0
        TTI_SFPADDI(0xcf00, p_sfpu::LREG0, 0);                            // lreg[0] += -2**31
        TTI_SFPENCC(0, 0, 0, 0);                                          // restore cc
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 4 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // Note: L0=0.0 and L1=-2**31.  The sign bit of abs(v) is stored in L7 and
    // used to pick L0 or L1 for SFPMAD's VA:
    //
    // - if sign bit is 0, then compute L0*1.0 + v = v
    // - if sign bit is 1, then compute L1*1.0 + v = -2**31 + 0.0 = -2**31
    //
    // t | Load | Simple             | MAD                 | Round            | Store   |
    // - | ---- | ------------------ | ------------------- | ---------------- | ------- |
    // 0 | [v]  |                    |                     |                  |         |
    // 1 |      | t = abs(v)         |                     |                  |         |
    // 2 |      |                    |                     | L7 = t >> 31     |         |
    // 3 |      | t = cast(t)        |                     |                  |         |
    // 0 | ...  | [v] = setsgn(t, v) |                     |                  |         |
    // 1 | ...  |                    | [v] = L[L7]*1.0 + v |                  |         |
    // 2 | ...  |                    |                     |                  |         |
    // 3 | ...  |                    |                     | [v] L16 = rnd(v) |         |
    // 0 | ...  |                    |                     |                  | [v] L16 |

    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);  // -2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        int v = 2 + (d & 1);  // alternate between p_sfpu::LREG2 and p_sfpu::LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5);  // SFPSHFT2_MOD1_SHFT_LREG
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_int32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        // result = 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        // exp = in.Exp (LaneEnabled = exp >= 0)
        TTI_SFPEXEXP(
            0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = INT_MIN
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
        // exp -= 31 (LaneEnabled = exp < 31)
        TTI_SFPIADD(-31 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 8
        TTI_SFPIADD(8, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = exman(in, sfpi::MantissaMode::ImplicitOne) << (exp - 23)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        // LaneEnabled = in < 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // result = -result (two's complement)
        TTI_SFPIADD(
            0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        // result = 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        // LaneEnabled = in >= 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
        // exp = in.Exp (LaneEnabled = exp >= 0)
        TTI_SFPEXEXP(
            0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // result = 0xffffffff
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
        // exp -= 32 (LaneEnabled = exp < 31)
        TTI_SFPIADD(-32 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 9
        TTI_SFPIADD(9, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = exman(in, sfpi::MantissaMode::ImplicitOne) << (exp - 23)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_fp16b() {
    // TODO: Attempt to use LOADMACRO #46751
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG1, p_sfpu::LREG0, 5);                // lreg[0] = lreg[1] >> 16
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);                            // lreg[0] &= 1
        TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);  // lreg[1] += 0x7FFF
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);   // lreg[1] += lreg[0]
        TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG1, 0);                            // lreg[1] &= 0xFFFF0000
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp32() {
    // TODO: Attempt to use LOADMACRO #46751
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp32() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);                   // lreg[1] = iabs(lreg[0])
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, 0);                     // lreg[2] = cast(lreg[1])
        TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);                // lreg[0] = sign(lreg[0]) | exp_man(lreg[2])
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // cc = lreg[1] < 0
        TTI_SFPADDI(0xcf00, p_sfpu::LREG0, 0);                            // lreg[0] += -2**31
        TTI_SFPENCC(0, 0, 0, 0);                                          // restore cc
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 4 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // Note: L0=0.0 and L1=-2**31.  The sign bit of abs(v) is stored in L7 and
    // used to pick L0 or L1 for SFPMAD's VA:
    //
    // - if sign bit is 0, then compute L0*1.0 + v = v
    // - if sign bit is 1, then compute L1*1.0 + v = -2**31 + 0.0 = -2**31
    //
    // t | Load | Simple             | MAD                     | Round        | Store   |
    // - | ---- | ------------------ | ----------------------- | ------------ | ------- |
    // 0 | [v]  |                    |                         |              |         |
    // 1 |      | t = abs(v)         |                         |              |         |
    // 2 |      |                    |                         | L7 = t >> 31 |         |
    // 3 |      | t = cast(t)        |                         |              |         |
    // 0 | ...  | [v] = setsgn(t, v) |                         |              |         |
    // 1 | ...  |                    | [v] L16 = L[L7]*1.0 + v |              |         |
    // 2 | ...  |                    |                         |              |         |
    // 3 | ...  |                    |                         |              | [v] L16 |

    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);  // -2**31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        int v = 2 + (d & 1);  // alternate between p_sfpu::LREG2 and p_sfpu::LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5);  // SFPSHFT2_MOD1_SHFT_LREG
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp16b() {
    // TODO: Attempt to use LOADMACRO #46751
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPADDI(0x4f00, p_sfpu::LREG1, 0);  // 2^31
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp32() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPADDI(0x4f00, p_sfpu::LREG2, 0);  // 2^31
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 3 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // Note: L0=0.0 and L1=2**31.  The sign bit is stored in L7 and used to pick L0 or L1
    // for SFPMAD's VA:
    //
    // - if sign bit is 0, then compute L0*1.0 + v = v
    // - if sign bit is 1, then compute L1*1.0 + v = 2**31 + v
    //
    // t | Load | Simple             | MAD                     | Round       | Store    |
    // - | ---- | ------------------ | ----------------------- | ----------- | -------- |
    // 0 | [a]  |                    |                         |             |          |
    // 1 | [b]  | [a] = setsgn(a, 0) |                         |             |          |
    // 2 | [L7] | [b] = cast(a)      |                         |             |          |
    // 0 | ...  |                    |                         | [L7] >>= 31 |          |
    // 1 | ...  |                    | [b] L16 = L[L7]*1.0 + b |             |          |
    // 2 | ...  |                    |                         |             |          |
    // 0 | ...  |                    |                         |             | [L7] L16 |

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // 2**31

    constexpr int a = p_sfpu::LREG2;
    constexpr int b = p_sfpu::LREG3;
    constexpr int L7 = p_sfpu::LREG7;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_7, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_7, b >> 2);
        TTI_SFPLOADMACRO((2 << 2) | (L7 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, L7 >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_uint32() {
    // TODO: Attempt to use LOADMACRO #46751
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint16() {
    // TODO: Attempt to use LOADMACRO #46751
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT((-16) & 0xFFF, 0, p_sfpu::LREG0, 1);
        TTI_SFPGT(0, p_sfpu::LCONST_0, p_sfpu::LREG0, SFPGT_MOD1_SET_ALL_ONES);  // Set LREG0 = -1 if greater than 0
        TTI_SFPOR(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);  // Leaves garbage in high bits, but packer will ignore it
        TTI_SFPSTORE(p_sfpu::LREG1, SFPSTORE_MODE_SWAP_HI_LO16, ADDR_MOD_6, 0);  // Swap hi and low 16 before write
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint16() {
    // TODO: Attempt to use LOADMACRO #46751
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 9);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
        TTI_SFPSTORE(p_sfpu::LREG0, SFPSTORE_MODE_SWAP_HI_LO16, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_fp16b() {
    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
    sfpi::vConstIntPrgm2 = 0xffff0000;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_uint32() {
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp32() {
#ifndef DISABLE_SFPLOADMACRO
    // SFPCAST interprets its input as sign-magnitude, so bit 31 of the source
    // flags the case that needs a post-cast fixup. vConstIntPrgm0 (LREG12) is
    // preloaded with -31 -- the shift amount used to extract that bit.
    sfpi::vConstIntPrgm0 = -31;

    constexpr int a = p_sfpu::LREG2;

    // InstructionTemplate[0]
    TTI_SFPSETSGN(0, 0, 12, 1);  // SFPSETSGN_MOD1_ARG_IMM

    // InstructionTemplate[1]
    TTI_SFPCAST(a, 13, 0);

    // InstructionTemplate[2]
    TTI_SFPSHFT2(0, p_sfpu::LREG12, 14, 5);  // SFPSHFT2_MOD1_SHFT_LREG

    // InstructionTemplate[3]
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 15, 4);  // SFPMAD_MOD1_INDIRECT_VA

    // Macro 0: [a]
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 0, 1);
    }
    // Macro 1: [b]
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits = 0x00 | 0x40 | (2 << 3) | (4 + 3);

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }
    // Macro 2: [L7]
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0x80 | 0x00 | (0 << 3) | (4 + 2);
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Misc: {
    //   StoreMod0: FP32,
    //   UsesLoadMod0ForStore: {0,0,0},
    //   UnitDelayKind: {1,1,1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x700 | InstrModLoadStore::FP32, 8, 1);
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp32() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int t = p_sfpu::LREG4;

    // SFPCAST interprets its input as sign-magnitude, so bit 31 of the source
    // flags the case that needs a post-cast fixup. vConstIntPrgm0 (LREG12) is
    // preloaded with -31 -- the shift amount used to extract that bit.
    sfpi::vConstIntPrgm0 = -31;

    // InstructionTemplate[0]
    TTI_SFPSETSGN(0, t, 12, 0);

    // InstructionTemplate[1]
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 13, 4);  // SFPMAD_MOD1_INDIRECT_VA

    // Macro 0: [v]
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (3 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0x00 | 0x40 | (4 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (6 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: FP32,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::FP32, 8, 1);
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp16b() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int t = p_sfpu::LREG4;

    // SFPCAST interprets its input as sign-magnitude, so bit 31 of the source
    // flags the case that needs a post-cast fixup. vConstIntPrgm0 (LREG12) is
    // preloaded with -31 -- the shift amount used to extract that bit.
    sfpi::vConstIntPrgm0 = -31;

    // InstructionTemplate[0]
    TTI_SFPSETSGN(0, t, 12, 0);

    // InstructionTemplate[1]
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 13, 4);  // SFPMAD_MOD1_INDIRECT_VA

    // InstructionTemplate[2]
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);

    // Macro 0: [v]
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (3 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0x00 | 0x00 | (4 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = 0x00 | 0x40 | (6 << 3) | (4 + 2);
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (7 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp32() {
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp16b() {
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPCAST(0, 12, 0);

    // InstructionTemplate[1]
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);

    // Macro 0
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0x00 | 0x40 | (1 << 3) | (4 + 1);
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: DEFAULT,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp16b() {}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint16() {}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_uint16() {}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_uint16() {}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        // result = 0 (default for zero, subnormals, and |in| < 1.0)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);
        // exponent = exexp(in); LaneEnabled = |in| >= 1.0
        // (CC flags avoid SFPEXEXP quirk: zero/subnormal biased_exp=0 returns wrong value)
        TTI_SFPEXEXP(
            0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // mantissa = exman(in, sfpi::MantissaMode::ImplicitOne)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // shift_amount = exponent - 23
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result = floor(|in|)
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = in < 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // result = -result  (two's complement negate)
        TTI_SFPIADD(
            0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);
        // result += 256 (packer format; for negatives: −|v|+256 gives correct uint8 wrap)
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // result &= 0xFF
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool u16 = false>
inline void calculate_typecast_uint_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        if constexpr (u16) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
            TTI_SFPAND(0, p_sfpu::LREG13, p_sfpu::LREG0, 0);
        } else {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        }
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    sfpi::vConstIntPrgm0 = 0xFF;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint_to_uint8() {
    sfpi::vConstIntPrgm0 = 0xFF;
    sfpi::vConstIntPrgm1 = UINT16_LOW_MASK;
}

}  // namespace sfpu
}  // namespace ckernel
