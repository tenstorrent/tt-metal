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
#include "ckernel_sfpu_quant.h"  // for INT8_SIGN_MASK
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Mask that keeps the low 16 bits (the UInt16 value) of a 32-bit dest word and clears the garbage high bits.
constexpr std::uint16_t UINT16_LOW_MASK = 0xFFFF;

// SFPSTORE mode that swaps the high and low 16 bits before writing, so a value computed in the low 16 bits
// lands in the high 16 bits where the packer reads UInt16 out of a 32-bit dest word.
constexpr std::uint32_t SFPSTORE_MODE_SWAP_HI_LO16 = 9;

// -128.0f as the upper 16 bits
constexpr std::uint32_t TYPECAST_INT8_MINUS_128_IMM16 = 0xC300;

// -128 as SFPIADD's 12-bit signed immediate
constexpr std::int32_t TYPECAST_INT8_MINUS_128_IMM12 = -128 & 0xfff;

// SFPGT mod1 selector that sets the destination to all-ones (-1) when the comparison is true.
constexpr std::uint32_t SFPGT_MOD1_SET_ALL_ONES = 8;

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_typecast_fp32_to_uint16() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 9);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
        if (is_fp32_dest_acc_en) {
            TTI_SFPSTORE(p_sfpu::LREG0, SFPSTORE_MODE_SWAP_HI_LO16, ADDR_MOD_6, 0);
        } else {
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_6, 0);
        }
    }
#else
    // SFPLOADMACRO fast path in both Dest modes, throughput of 2 cycles per input row; init_typecast_fp32_to_uint16
    // programs the macro's store for the Dest mode (LO16 into a 16-bit Dest, swap-hi-lo16 into a 32-bit one).
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple            | MAD | Round            | Store   |
    // - | ---- | ----------------- | --- | ---------------- | ------- |
    // 0 | [v]  |                   |     |                  |         |
    // 1 | nop  | [v] = max(v, 0.0) |     |                  |         |
    // 0 | ...  | (must be idle)    |     | (must be idle)   |         |
    // 1 | ...  |                   |     | [v] L16 = rnd(v) |         |
    // 0 | ...  |                   |     |                  | [v] L16 |

    // SFPLOADMACRO operand encoding: operand0 = (macro_select << 2) | (VD & 3) and the
    // trailing operand = VD >> 2, so the hardware reconstructs the value-register index
    // VD = (trailing << 2) | (operand0 & 3) -- a 3-bit index spanning LREG0..LREG7 -- while
    // operand0[3:2] selects which armed macro fires. Here VD is 0/1 and macro_select 0, so
    // the mask/shift are no-ops, but the same idiom addresses VD >= 4 elsewhere (e.g.
    // calculate_typecast_uint32_to_fp32 fires macro 2 with VD = LREG7).
    // Two rows per trip, so the macro words are compile-time constants at any row count.
#pragma GCC unroll 8
    for (int d = 0; d + 1 < ITERATIONS; d += 2) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, p_sfpu::LREG0 >> 2);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG1 & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, p_sfpu::LREG1 >> 2);
        TTI_SFPNOP;
    }
    if constexpr (ITERATIONS & 1) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, p_sfpu::LREG0 >> 2);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
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

    // Two rows per trip, so the macro words are compile-time constants at any row count.
#pragma GCC unroll 8
    for (int d = 0; d + 1 < ITERATIONS; d += 2) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::LO16, ADDR_MOD_6, p_sfpu::LREG0 >> 2);
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG1 & 3), InstrModLoadStore::LO16, ADDR_MOD_6, p_sfpu::LREG1 >> 2);
    }
    if constexpr (ITERATIONS & 1) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::LO16, ADDR_MOD_6, p_sfpu::LREG0 >> 2);
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

    // Two rows per trip, so the macro words are compile-time constants at any row count.
#pragma GCC unroll 8
    for (int d = 0; d + 1 < ITERATIONS; d += 2) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG2 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG2 >> 2);
        TTI_SFPABS(0, p_sfpu::LREG2, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
        TTI_SFPCAST(t, t, 0);
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG3 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG3 >> 2);
        TTI_SFPABS(0, p_sfpu::LREG3, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
        TTI_SFPCAST(t, t, 0);
    }
    if constexpr (ITERATIONS & 1) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG2 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG2 >> 2);
        TTI_SFPABS(0, p_sfpu::LREG2, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
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
    // Truncation toward zero, saturation to INT32_MAX / INT32_MIN by the sign (NaN too), 0 for |in| < 1. The sign
    // is folded as (m ^ s) - s with s = in >> 31; INT32_MAX comes from vConstIntPrgm0 (init_typecast_fp32_to_int32).
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        // result = 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);
        // s = in >> 31 (arithmetic shift: SFPSHFT mod1 = ARG_IMM | ARITHMETIC | ARG_IMM_USE_VC)
        TTI_SFPSHFT(-31 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG4, 7);

        // exp = in.Exp (LaneEnabled = exp >= 0)
        TTI_SFPEXEXP(
            0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // shift = exp - 23
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // exp -= 31 (LaneEnabled = exp < 31)
        TTI_SFPIADD(-31 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // result = exman(in, sfpi::MantissaMode::ImplicitOne) << shift
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);

        // result ^= s
        TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG1, 0);
        // LaneEnabled = exp - 31 >= 0: the saturating lanes (|in| >= 2^31, inf, NaN) take INT32_MAX (LREG12)
        TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
        TTI_SFPMOV(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);
        // s = result - s, the second half of the sign fold
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
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
    // Kept as a plain loop (no SFPLOADMACRO): #46231 rewrote this to round, mask off the
    // low 16 bits (&0xFFFF0000), and store FP32 for 32-bit-Dest correctness. The historical
    // macro relied on a FP16B store to truncate and does not reproduce the masked-FP32 result,
    // so it is not equivalent and is not restored here. See #46751.
#pragma GCC unroll 8
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

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_typecast_uint16_to_fp32() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
#else
    if constexpr (!is_fp32_dest_acc_en) {
        // 16-bit Dest: SFPLOADMACRO fast path, throughput of 1 cycle per input row. The LO16 load
        // keeps only the low 16 bits (the UInt16 value), so casting it matches the plain loop's
        // INT32 load + 0xFFFF mask + cast.
        //
        // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
        //
        // t | Load | Simple            | MAD | Round | Store   |
        // - | ---- | ----------------- | --- | ----- | ------- |
        // 0 | [v]  |                   |     |       |         |
        // 0 | ...  | [v] L16 = cast(v) |     |       |         |
        // 0 | ...  |                   |     |       | [v] L16 |

        constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);
        }
        TTI_SFPNOP;
        TTI_SFPNOP;
    } else {
        // 32-bit Dest: macro 1 (init_typecast_uint16_to_fp32) loads the INT32 word, masks it with LREG1 at delay 1
        // and stores it as FP32 at delay 3; the explicit SFPCAST of the row before fills the slot in between.
        static_assert(ITERATIONS % 4 == 0, "the four-register rotation takes a multiple of four rows");
        constexpr int r0 = p_sfpu::LREG0;
        constexpr int r1 = p_sfpu::LREG2;
        constexpr int r2 = p_sfpu::LREG3;
        constexpr int r3 = p_sfpu::LREG4;

        TTI_SFPLOADMACRO((1 << 2) | (r0 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, r0 >> 2);
        TTI_SFPNOP;
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d += 4) {
            // loads of rows d+1 .. d+4 interleaved with the casts of rows d .. d+3
            TTI_SFPLOADMACRO((1 << 2) | (r1 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, r1 >> 2);
            TTI_SFPCAST(r0, r0, 0);
            TTI_SFPLOADMACRO((1 << 2) | (r2 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, r2 >> 2);
            TTI_SFPCAST(r1, r1, 0);
            TTI_SFPLOADMACRO((1 << 2) | (r3 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, r3 >> 2);
            TTI_SFPCAST(r2, r2, 0);
            if (d + 4 < ITERATIONS) {
                TTI_SFPLOADMACRO((1 << 2) | (r0 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, r0 >> 2);
            } else {
                TTI_SFPNOP;
            }
            TTI_SFPCAST(r3, r3, 0);
        }
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
#endif
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

    // Two rows per trip, so the macro words are compile-time constants at any row count.
#pragma GCC unroll 8
    for (int d = 0; d + 1 < ITERATIONS; d += 2) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG2 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG2 >> 2);
        TTI_SFPABS(0, p_sfpu::LREG2, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
        TTI_SFPCAST(t, t, 0);
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG3 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG3 >> 2);
        TTI_SFPABS(0, p_sfpu::LREG3, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
        TTI_SFPCAST(t, t, 0);
    }
    if constexpr (ITERATIONS & 1) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG2 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG2 >> 2);
        TTI_SFPABS(0, p_sfpu::LREG2, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
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
#ifdef DISABLE_SFPLOADMACRO
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
    // t | Load | Simple             | MAD                     | Round            | Store   |
    // - | ---- | ------------------ | ----------------------- | ---------------- | ------- |
    // 0 | [v]  |                    |                         |                  |         |
    // 1 |      |                    |                         | L7 = v >> 31     |         |
    // 2 |      | v = setsgn(v, 0)   |                         |                  |         |
    // 0 | ...  | [v] = cast(v)      |                         |                  |         |
    // 1 | ...  |                    | [v] v = L[L7]*1.0 + v   |                  |         |
    // 2 | ...  |                    |                         |                  |         |
    // 0 | ...  |                    |                         | [v] L16 = rnd(v) |         |
    // 1 | ...  |                    |                         |                  | [v] L16 |

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // 2**31

    // Two rows per trip, so the macro words are compile-time constants at any row count.
#pragma GCC unroll 8
    for (int d = 0; d + 1 < ITERATIONS; d += 2) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG2 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG2 >> 2);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
        TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG2, 1);
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG3 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG3 >> 2);
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    }
    if constexpr (ITERATIONS & 1) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG2 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG2 >> 2);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
        TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG2, 1);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp32() {
    // Split x = high * 2^23 + low. Both high - 1 and 2^23 + low are exact FP32
    // values, so (high - 1) * 2^23 + (2^23 + low) rounds only once.
    // L12 = -23; L13 = 2^23. SFPSETMAN constructs 2^23 + low directly.
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPADDI(0xbf80, p_sfpu::LREG0, 0);  // -1.0
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPSETMAN(0, p_sfpu::LREG13, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
#else
    // Three issued instructions per row. Alternate register pairs so the next
    // row can start before the previous row's MAD and store finish.
    //
    // t | Load | Simple             | MAD                | Round    | Store |
    // 0 | h    |                    |                    |          |       |
    // 1 | l    |                    |                    | h >>= 23 |       |
    // 2 |      | h = cast(h)        |                    |          |       |
    // 3 | next | l = setman(L13, l) | h += -1.0          |          |       |
    // 5 |      |                    | l = h * L13 + l    |          |       |
    // 7 |      |                    |                    |          | l     |
    // The final MAD is issued explicitly; the remaining operations are macros.
    // Row 0, then two rows per trip, so the macro words are compile-time constants at any row count.
    if constexpr (ITERATIONS > 0) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, p_sfpu::LREG0 >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (p_sfpu::LREG2 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG2 >> 2);
        TTI_SFPNOP;
    }
#pragma GCC unroll 8
    for (int d = 1; d + 1 < ITERATIONS; d += 2) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, p_sfpu::LREG1 >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (p_sfpu::LREG3 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG3 >> 2);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, p_sfpu::LREG2, 0);
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, p_sfpu::LREG0 >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (p_sfpu::LREG2 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG2 >> 2);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, p_sfpu::LREG3, 0);
    }
    if constexpr (ITERATIONS > 1 && (ITERATIONS & 1) == 0) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, p_sfpu::LREG1 >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (p_sfpu::LREG3 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG3 >> 2);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, p_sfpu::LREG2, 0);
    }
    if constexpr (ITERATIONS > 0) {
        TTI_SFPNOP;
        TTI_SFPNOP;
        constexpr int h = (ITERATIONS - 1) & 1;
        constexpr int l = 2 + ((ITERATIONS - 1) & 1);
        TTI_SFPMAD(h, p_sfpu::LREG13, l, l, 0);
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_typecast_uint16_to_uint32() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
#else
    if constexpr (!is_fp32_dest_acc_en) {
        // 16-bit Dest: SFPLOADMACRO fast path, throughput of 1 cycle per input row. The LO16 load
        // keeps only the low 16 bits (the UInt16 value) and zero-extends them, so the INT32 store
        // matches the plain loop's INT32 load + 0xFFFF mask.
        //
        // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
        //
        // t | Load | Simple | MAD | Round | Store |
        // - | ---- | ------ | --- | ----- | ----- |
        // 0 | [v]  |        |     |       |       |
        // 0 | ...  |        |     |       | [v]   |

        constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);
        }
        TTI_SFPNOP;
    } else {
        // 32-bit Dest: macro 1 (init_typecast_uint16_to_uint32) loads the INT32 word, masks it with LREG1 and stores
        // it from LREG16, one issue slot per row.

        constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOADMACRO((1 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2);
        }
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint16() {
    // Kept as a plain loop (no SFPLOADMACRO): #46231 rewrote this to shift the value right by 16
    // and saturate via SFPGT on the *high* bits before the swap-hi-lo16 store. The historical
    // macro tested the low 16 bits instead and computes a different result, so it is not
    // equivalent and is not restored here. See #46751.
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
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 9);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
        TTI_SFPSTORE(p_sfpu::LREG0, SFPSTORE_MODE_SWAP_HI_LO16, ADDR_MOD_6, 0);
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 3 cycles per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple            | MAD | Round            | Store   |
    // - | ---- | ----------------- | --- | ---------------- | ------- |
    // 0 | [a]  |                   |     |                  |         |
    // 1 |      | a = cast_fp32(a)  |     |                  |         |
    // 2 | nop  | [a] = max(0.0, a) |     |                  |         |
    // 0 | ...  | (must be idle)    |     |                  |         |
    // 1 | ...  |                   |     | [a] L16 = rnd(a) |         |
    // 2 | ...  |                   |     |                  | [a] swap|
    //
    // Simple/Round sub-units can be used simultaneously if one has VD=16 and
    // the other VD!=16.  The following steps clamp the input value to 0-65535:
    //
    // a = cast_fp32(a); this allows us to use SFPSTOCHRND later to convert to uint16, clamping to 65535.
    // swap_minmax(0.0, a); since SFPSTOCHRND takes the absolute value before clamping, we use SFPSWAP to clamp negative
    // values to 0.0. L16 = rnd(a); finally, we use SFPSTOCHRND to clamp large values to 65535, using VD=16. The macro
    // Store uses SFPSTORE_MODE_SWAP_HI_LO16, matching the plain-loop store that lands the uint16 in the high 16 bits.

    // Two rows per trip, so the macro words are compile-time constants at any row count.
#pragma GCC unroll 8
    for (int d = 0; d + 1 < ITERATIONS; d += 2) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG0 >> 2);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG1 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG1 >> 2);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPNOP;
    }
    if constexpr (ITERATIONS & 1) {
        TTI_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, p_sfpu::LREG0 >> 2);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_fp16b() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
    sfpi::vConstIntPrgm2 = 0xffff0000;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_int32() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    // INT32_MAX for the saturating lanes of calculate_typecast_fp32_to_int32.
    sfpi::vConstIntPrgm0 = std::numeric_limits<std::int32_t>::max();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_uint32() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifdef DISABLE_SFPLOADMACRO
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);
#else
    // LREG1 holds the mask of the 32-bit Dest path of calculate_typecast_uint16_to_uint32 (macro 1 below); the
    // macro programming only targets LREG0, so it survives. The 16-bit Dest macro path does not read LREG1.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);

    // InstructionTemplate[0]: the mask of macro 1 (32-bit Dest); the loaded register replaces VB, LREG16 VD
    TTI_SFPAND(0, p_sfpu::LREG1, 12, 0);

    // Macro 0 (16-bit Dest): store only (the LO16 load already zero-extends the UInt16 value).
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (0 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1 (32-bit Dest): mask into LREG16 at delay 0, INT32 store of LREG16 at delay 1.
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x40 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (1 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Misc: {
    //   StoreMod0: INT32,
    //   UsesLoadMod0ForStore: {0,0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
#endif
}

// SFPCAST interprets its input as sign-magnitude, so bit 31 of the source flags the case
// that needs a post-cast fixup. vConstIntPrgm0 (LREG12) is preloaded with -31 -- the shift
// amount used by init_typecast_int32_to_fp32 and init_typecast_{uint32,int32}_to_fp16b
// to extract that bit.
inline void preload_sign_magnitude_cast_fixup() { sfpi::vConstIntPrgm0 = -31; }

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp32() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstIntPrgm0 = -23;
    sfpi::vConstFloatPrgm1 = 8388608.0f;  // 2^23
#ifndef DISABLE_SFPLOADMACRO
    // Instruction templates: shift, cast, subtract 1.0, construct 2^23 + low.
    TTI_SFPSHFT2(0, p_sfpu::LREG12, 12, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPCAST(0, 13, 0);
    TTI_SFPADDI(0xbf80, 14, 0);
    TTI_SFPSETMAN(0, p_sfpu::LREG13, 15, 0);

    // A disabled unit must also use delay 7, otherwise it can cancel a pending
    // instruction from an earlier load macro at the same delay.
    constexpr std::uint32_t disabled = 7 << 3;
    // Macro 0: high = fp32(x >> 23) - 1.0.
    {
        constexpr std::uint32_t simple_bits = (1 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits = (2 << 3) | (4 + 2);
        constexpr std::uint32_t round_bits = 0x80 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // Macro 1: low = 2^23 + (x & 0x7fffff); store after the explicit MAD.
    {
        constexpr std::uint32_t simple_bits = 0x80 | (1 << 3) | (4 + 3);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = (5 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    // FP32 stores; all units wait for issued SFPU instructions, so a stalled
    // instruction stream cannot let the store overtake the explicit MAD.
    TTI_SFPCONFIG(0xf00 | InstrModLoadStore::FP32, 8, 1);
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp32() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    constexpr int t = p_sfpu::LREG4;

    preload_sign_magnitude_cast_fixup();

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
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    constexpr int t = p_sfpu::LREG4;

    preload_sign_magnitude_cast_fixup();

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
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifdef DISABLE_SFPLOADMACRO
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);
#else
    // LREG1 holds the mask of the 32-bit Dest path of calculate_typecast_uint16_to_fp32 (macro 1 below); the
    // macro programming only targets LREG0, so it survives. The 16-bit Dest macro path does not read LREG1.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);

    // InstructionTemplate[0]: the cast (16-bit Dest, macro 0)
    TTI_SFPCAST(0, 12, 0);

    // InstructionTemplate[1]: the mask of macro 1 (32-bit Dest); the loaded register replaces VB and VD
    TTI_SFPAND(0, p_sfpu::LREG1, 13, 0);

    // Macro 0 (16-bit Dest): cast into LREG16 at delay 0, FP32 store of LREG16 at delay 1
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (1 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1 (32-bit Dest): mask in place at delay 1, FP32 store at delay 3; the body casts in between.
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (1 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Misc: {
    //   StoreMod0: FP32,
    //   UsesLoadMod0ForStore: {0,0},
    //   UnitDelayKind: {1,0,0,1}, (WaitForElapsedInstructions=1 for the Simple and Store sub-units)
    // }
    TTI_SFPCONFIG(0x900 | InstrModLoadStore::FP32, 8, 1);
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp16b() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
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
inline void init_typecast_uint32_to_fp16b() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    preload_sign_magnitude_cast_fixup();

    // InstructionTemplate[0]
    TTI_SFPCAST(0, 12, 0);

    // InstructionTemplate[1]
    TTI_SFPMAD(0, p_sfpu::LCONST_1, 0, 13, 4);  // SFPMAD_MOD1_INDIRECT_VA

    // InstructionTemplate[2]
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);

    // Macro 0
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (2 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0x00 | 0x00 | (3 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = 0x00 | 0x40 | (5 << 3) | (4 + 2);
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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void init_typecast_fp32_to_uint16() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    // Programs the macro of calculate_typecast_fp32_to_uint16 for both Dest modes; only the store mode follows the
    // Dest mode: LO16 into a 16-bit Dest, SFPSTORE_MODE_SWAP_HI_LO16 (high half of the word) into a 32-bit Dest.

    // InstructionTemplate[0]
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, 12, 0xf);  // L[VD] = max(0, L[VD])

    // InstructionTemplate[1]
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);

    // Macro 0
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0x00 | 0x40 | (2 << 3) | (4 + 1);
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: LO16 (16-bit Dest) or SFPSTORE_MODE_SWAP_HI_LO16 (32-bit Dest),
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    constexpr std::uint32_t store_mode =
        is_fp32_dest_acc_en ? SFPSTORE_MODE_SWAP_HI_LO16 : static_cast<std::uint32_t>(InstrModLoadStore::LO16);
    TTI_SFPCONFIG(0x100 | store_mode, 8, 1);
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_uint16() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_uint16() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, 12, 0xf);  // L[VD] = max(0, L[VD])

    // InstructionTemplate[1]
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);

    // Macro 0
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (1 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0x00 | 0x40 | (3 << 3) | (4 + 1);
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (4 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: SFPSTORE_MODE_SWAP_HI_LO16 (swap hi/lo 16 before write),
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | SFPSTORE_MODE_SWAP_HI_LO16, 8, 1);
#endif
}

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

template <bool APPROXIMATION_MODE, int ITERATIONS, bool CLAMP_TO_UINT16 = false>
inline void calculate_typecast_int8_to_int32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);  // e = b ^ 0x80 to get excess 128
        if constexpr (CLAMP_TO_UINT16) {
            TTI_SFPIADD(
                TYPECAST_INT8_MINUS_128_IMM12,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
            TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);  // negatives clamp to 0
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPSTORE(p_sfpu::LREG0, SFPSTORE_MODE_SWAP_HI_LO16, ADDR_MOD_6, 0);
        } else {
            TTI_SFPIADD(
                TYPECAST_INT8_MINUS_128_IMM12,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
        }
    }
}

// Also serves the Float16_b/Bfp8_b/Bfp4_b outputs. Those need no FP32_TO_FP16B round before the
// store the way the uint path does, because every value here is in [-128, 127] and so is exact in
// bfloat16's 8-bit significand.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int8_to_fp32() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);  // e = b ^ 0x80 in [0, 255]
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPADDI(TYPECAST_INT8_MINUS_128_IMM16, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
#else
    // This uses SFPLOADMACRO to achieve a throughput of 3 issue slots per input row. The XOR -> CAST
    // order is forced (SFPCAST reads sign-magnitude), so only the scheduling changes: the macros
    // below hold the same XOR, CAST and subtract, spread across the Simple, MAD and Store sub-units
    // so consecutive rows overlap.
    //
    // Three macros is not a requirement: macro 2 carries only the Store, and its load is dead,
    // unlike in init_typecast_uint32_to_fp32 where the MAD reads the loaded value back through an
    // indirect VA. Folding the Store onto macro 1 would cut this to 2 slots per row, at the cost of
    // leaning on the minimum MAD-to-Store delay. Left alone because the gain would not show up:
    // even moving from the plain loop to macros only helped at 32x32, with larger shapes
    // bandwidth-bound.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // Note: L0=-128.0, added by MAD to undo the excess 128 bias the XOR applied to a.
    //
    // t | Load | Simple         | MAD              | Round | Store    |
    // - | ---- | -------------- | ---------------- | ----- | -------- |
    // 0 | [a]  |                |                  |       |          |
    // 1 | [b]  | [a] = a ^ 0x80 |                  |       |          |
    // 2 | [L7] | [b] = cast(a)  |                  |       |          |
    // 0 | ...  |                |                  |       |          |
    // 1 | ...  |                | [b] L16 = L0 + b |       |          |
    // 2 | ...  |                |                  |       |          |
    // 0 | ...  |                |                  |       | [L7] L16 |

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, TYPECAST_INT8_MINUS_128_IMM16);

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

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstIntPrgm0 = 0xFF;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint_to_uint8() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstIntPrgm0 = 0xFF;
    sfpi::vConstIntPrgm1 = UINT16_LOW_MASK;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int8_input() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstIntPrgm0 = INT8_SIGN_MASK;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int8_to_fp32() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstIntPrgm0 = INT8_SIGN_MASK;
#ifndef DISABLE_SFPLOADMACRO
    constexpr int a = p_sfpu::LREG2;

    // InstructionTemplate[0]
    TTI_SFPXOR(0, p_sfpu::LREG12, 12, 0);

    // InstructionTemplate[1]
    TTI_SFPCAST(a, 13, 0);

    // InstructionTemplate[2]
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, 0, 14, 0);

    // Macro 0: [a]
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 0, 1);
    }
    // Macro 1: [b]
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits = 0x00 | 0x40 | (2 << 3) | (4 + 2);

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }
    // Macro 2: [L7]
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
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

}  // namespace sfpu
}  // namespace ckernel
