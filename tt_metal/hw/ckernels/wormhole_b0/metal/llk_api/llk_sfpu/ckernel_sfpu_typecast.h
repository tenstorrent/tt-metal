// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Mask that keeps the low 16 bits (the UInt16 value) of a 32-bit dest word and clears the garbage high bits.
constexpr std::uint16_t UINT16_LOW_MASK = 0xFFFF;

// SFPSTORE mode that swaps the high and low 16 bits before writing, so a value computed in the low 16 bits
// lands in the high 16 bits where the packer reads UInt16 out of a 32-bit dest word.
constexpr std::uint32_t SFPSTORE_MODE_SWAP_HI_LO16 = 9;

// Disarms the SFPLOADMACRO "Misc" / Load-Macro-Control config that the typecast init functions program
// unconditionally (the init dispatch passes only APPROX, so it cannot see DST_ACCUM_MODE and always arms
// the macro). The 32-bit Dest (DST_ACCUM_MODE) typecast paths fall back to a plain TTI_ loop that never
// issues an SFPLOADMACRO. On Wormhole, leaving the Misc word armed with UnitDelayKind=WaitForElapsedInstructions
// and then running plain SFP instructions hard-hangs the SFPU (all three Tensix threads enter but never
// complete) -- see #46751. Blackhole tolerates the un-consumed macro state, so this is a WH-specific quirk.
// A full overwrite of the Load-Macro-Control register (config_dest=8) with imm16=0 zeroes StoreMod0 and
// clears UnitDelayKind, removing the hazard. The plain loop sets its own per-instruction store mode, so this
// is correctness-neutral. The two SFPNOPs let the SFPCONFIG settle (matching the topk restore idiom).
inline void disarm_sfploadmacro_misc() {
#ifndef DISABLE_SFPLOADMACRO
    TTI_SFPCONFIG(0x000, 8, 1);  // imm16=0, config_dest=8 (Load Macro Control / Misc), mod=1 (immediate overwrite)
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool DST_ACCUM_MODE>
inline void calculate_typecast_fp32_to_uint16() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 9);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
        if (DST_ACCUM_MODE) {
            TTI_SFPSTORE(p_sfpu::LREG0, SFPSTORE_MODE_SWAP_HI_LO16, ADDR_MOD_2, 0);
        } else {
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_2, 0);
        }
    }
#else
    if constexpr (!DST_ACCUM_MODE) {
        // 16-bit Dest: SFPLOADMACRO fast path, throughput of 2 cycles per input row.
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
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            int v = d & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
            TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, v >> 2);
            TTI_SFPNOP;
        }
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
    } else {
        // 32-bit Dest: the swap-hi-lo16 store cannot be expressed by the init-time macro store
        // mode, so this case uses the plain loop. The init still armed the macro Misc word, so
        // disarm the leftover state first (WH hangs otherwise running a plain loop with it, #46751).
        disarm_sfploadmacro_misc();
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
            TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 9);
            TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
            TTI_SFPSTORE(p_sfpu::LREG0, SFPSTORE_MODE_SWAP_HI_LO16, ADDR_MOD_2, 0);
        }
    }
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp16b() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
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
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
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
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);                   // lreg[1] = iabs(lreg[0])
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, 0);                     // lreg[2] = cast(lreg[1])
        TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);                // lreg[0] = sign(lreg[0]) | exp_man(lreg[2])
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // cc = lreg[1] < 0
        TTI_SFPADDI(0xcf00, p_sfpu::LREG0, 0);                            // lreg[0] += -2**31
        TTI_SFPENCC(0, 0, 0, 0);                                          // restore cc
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
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
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPABS(0, v, t, 0);
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
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
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

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
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

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_fp16b() {
    // Kept as a plain loop (no SFPLOADMACRO): #46231 rewrote this to round, mask off the
    // low 16 bits (&0xFFFF0000), and store FP32 for 32-bit-Dest correctness. The historical
    // macro relied on a FP16B store to truncate and does not reproduce the masked-FP32 result,
    // so it is not equivalent and is not restored here. See #46751.
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPSHFT((-16) & 0xFFF, 0, p_sfpu::LREG0, 1);                            // lreg[0] >>= 16
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);                            // lreg[0] &= 1
        TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);  // lreg[1] += 0x7FFF
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);   // lreg[1] += lreg[0]
        TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG1, 0);                            // lreg[1] &= 0xFFFF0000
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_2, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool DST_ACCUM_MODE>
inline void calculate_typecast_uint16_to_fp32() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_2, 0);
    }
#else
    if constexpr (!DST_ACCUM_MODE) {
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
            TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
        }
        TTI_SFPNOP;
        TTI_SFPNOP;
    } else {
        // 32-bit Dest: the macro's LO16 load cannot reproduce the INT32 + 0xFFFF mask path, so
        // this case uses the plain loop. The init still armed the macro Misc word, so disarm the
        // leftover state first (WH hangs otherwise running a plain loop with it, #46751).
        disarm_sfploadmacro_misc();
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
            TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_2, 0);
        }
    }
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp32() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);                   // lreg[1] = iabs(lreg[0])
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, 0);                     // lreg[2] = cast(lreg[1])
        TTI_SFPSETSGN(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);                // lreg[0] = sign(lreg[0]) | exp_man(lreg[2])
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // cc = lreg[1] < 0
        TTI_SFPADDI(0xcf00, p_sfpu::LREG0, 0);                            // lreg[0] += -2**31
        TTI_SFPENCC(0, 0, 0, 0);                                          // restore cc
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
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
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPABS(0, v, t, 0);
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
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPADDI(0x4f00, p_sfpu::LREG1, 0);  // 2^31
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_2, 0);
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

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        int v = 2 + (d & 1);  // alternate between p_sfpu::LREG2 and p_sfpu::LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPSHFT2(v, p_sfpu::LREG12, p_sfpu::LREG7, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
        TT_SFPSETSGN(0, v, v, 1);
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
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG2, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPADDI(0x4f00, p_sfpu::LREG2, 0);  // 2^31
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_2, 0);
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
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_3, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_3, b >> 2);
        TTI_SFPLOADMACRO((2 << 2) | (L7 & 3), InstrModLoadStore::INT32, ADDR_MOD_2, L7 >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool DST_ACCUM_MODE>
inline void calculate_typecast_uint16_to_uint32() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
#else
    if constexpr (!DST_ACCUM_MODE) {
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
            TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
        }
        TTI_SFPNOP;
    } else {
        // 32-bit Dest: the macro's LO16 load cannot reproduce the INT32 + 0xFFFF mask path, so
        // this case uses the plain loop. The init still armed the macro Misc word, so disarm the
        // leftover state first (WH hangs otherwise running a plain loop with it, #46751).
        disarm_sfploadmacro_misc();
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
            TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
        }
    }
#endif
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint16() {
    // Kept as a plain loop (no SFPLOADMACRO): #46231 rewrote this to shift the value right by 16
    // and renormalize (2's-complement + shift + OR) before the swap-hi-lo16 store. The historical
    // macro used a LO16-load 2-macro pipeline on a different bit layout and computes a different
    // result, so it is not equivalent and is not restored here. See #46751.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT((-16) & 0xFFF, 0, p_sfpu::LREG0, 1);
        TTI_SFPIADD(
            0, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE | sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST);
        TTI_SFPSHFT((-16) & 0xFFF, 0, p_sfpu::LREG0, 1);
        TTI_SFPOR(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, SFPSTORE_MODE_SWAP_HI_LO16, ADDR_MOD_2, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint16() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 9);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
        TTI_SFPSTORE(p_sfpu::LREG0, SFPSTORE_MODE_SWAP_HI_LO16, ADDR_MOD_2, 0);
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

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        int a = d & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_2, a >> 2);
        TT_SFPCAST(a, a, 0);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_fp16b() {
    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
    sfpi::vConstIntPrgm2 = 0xffff0000;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_uint32() {
#ifdef DISABLE_SFPLOADMACRO
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);
#else
    // The 32-bit Dest (DST_ACCUM_MODE) path of calculate_typecast_uint16_to_uint32 falls
    // back to the plain loop, which masks the loaded word with LREG1. Load the mask here
    // so that path is correct; the macro-programming below only targets LREG0, so LREG1
    // survives. The 16-bit Dest macro path does not read LREG1. The init cannot see
    // DST_ACCUM_MODE (the dispatch passes only APPROX), so the 32-bit Dest calc disarms the
    // macro before its plain loop (see disarm_sfploadmacro_misc / #46751).
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);

    // Macro 0: store only (the LO16 load already zero-extends the UInt16 value).
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (0 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: INT32,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
#endif
}

// SFPCAST interprets its input as sign-magnitude, so bit 31 of the source flags the case
// that needs a post-cast fixup. vConstIntPrgm0 (LREG12) is preloaded with -31 -- the shift
// amount the int/uint -> float macro inits (init_typecast_{uint32,int32}_to_fp32 / _to_fp16b)
// use to extract that bit.
inline void preload_sign_magnitude_cast_fixup() { sfpi::vConstIntPrgm0 = -31; }

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp32() {
#ifndef DISABLE_SFPLOADMACRO
    preload_sign_magnitude_cast_fixup();

    constexpr int a = p_sfpu::LREG2;

    // InstructionTemplate[0]
    TTI_SFPSETSGN(0, 0, 12, 1);  // SFPSETSGN_MOD1_ARG_IMM

    // InstructionTemplate[1]
    TTI_SFPCAST(a, 13, 0);

    // InstructionTemplate[2]
    TTI_SFPSHFT2(0, p_sfpu::LREG12, 14, sfpi::SFPSHFT2_MOD1_SHFT_LREG);

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
#ifdef DISABLE_SFPLOADMACRO
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);
#else
    // The 32-bit Dest (DST_ACCUM_MODE) path of calculate_typecast_uint16_to_fp32 falls
    // back to the plain loop, which masks the loaded word with LREG1. Load the mask here
    // so that path is correct; the macro-programming below only targets LREG0, so LREG1
    // survives. The 16-bit Dest macro path does not read LREG1. The init cannot see
    // DST_ACCUM_MODE (the dispatch passes only APPROX), so the 32-bit Dest calc disarms the
    // macro before its plain loop (see disarm_sfploadmacro_misc / #46751).
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);

    // InstructionTemplate[0]
    TTI_SFPCAST(0, 12, 0);

    // Macro 0
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (1 << 3) | 3;

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
inline void init_typecast_uint32_to_fp16b() {
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
    //   StoreMod0: FP32,  (changed from the historical DEFAULT to match the plain loop's FP32 store)
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::FP32, 8, 1);
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint16() {
#ifndef DISABLE_SFPLOADMACRO
    // Programs the macro used by the 16-bit Dest (LO16 store) path of
    // calculate_typecast_fp32_to_uint16. The 32-bit Dest path uses the plain loop and
    // disarms this macro first (the init cannot see DST_ACCUM_MODE; see #46751).

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
    //   StoreMod0: LO16,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::LO16, 8, 1);
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_uint16() {}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_uint16() {
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
    //   StoreMod0: SFPSTORE_MODE_SWAP_HI_LO16 (swap hi/lo 16 before write; changed from the
    //              historical LO16 to match the plain loop's swap-hi-lo16 store),
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
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
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
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool u16 = false>
inline void calculate_typecast_uint_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        if constexpr (u16) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
            TTI_SFPAND(0, p_sfpu::LREG13, p_sfpu::LREG0, 0);
        } else {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        }
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
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
