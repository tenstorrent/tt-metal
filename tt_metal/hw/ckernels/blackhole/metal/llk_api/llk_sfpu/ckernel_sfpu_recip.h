// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_rsqrt_compat.h"
#include "lltt.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Computes the reciprocal of a floating point value x.
template <int max_iter = 2>
sfpi_inline sfpi::vFloat sfpu_reciprocal_iter(const sfpi::vFloat x) {
    // sfpi::approx_recip(x) will return ±0 for x = ±inf or x ≥ ±2**126, and ±inf for x = ±0.
    sfpi::vFloat y = sfpi::approx_recip(x);

    // Optionally improve the approximation using Newton-Raphson.
    if constexpr (max_iter > 0) {
        // Normally, t = 2.0 - x * y, but we negate this (and negate again using y = y * -t later).
        // On Blackhole, when x=0 and y=infinity (and vice versa), t=+NaN regardless of the operand signs.
        // Negating the meaning of t makes it easier to detect NaN using a trivial sign check t>=0.
        // Equivalently, we could use v_if (t >= 2.0) instead, but SFPI doesn't support SFPLE/SFPGT at the moment.
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;

        if constexpr (max_iter > 1) {
            sfpi::vFloat y1 = y * -t - 0.0f;
            // If t=NaN, then t>=0.  This check consumes the SFPNOP slot of the preceding SFPMAD.
            v_if(t < 0) {
                t = x * y1 - sfpi::vConstFloatPrgm0;
                y = y1 * -t - 0.0f;
            }
            v_endif;
        } else {
            // If t=NaN, then t>=0.  This check cannot be hidden in a SFPNOP slot as it depends on the result of the
            // preceding SFPMAD.
            v_if(t < 0) { y = y * -t - 0.0f; }
            v_endif;
        }
    }

    return y;
}

// Approximate reciprocal, with throughput of 1c/32.
inline void _calculate_reciprocal_fast_7b_(const int iterations) {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
#else
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_6, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

// BF16 reciprocal using a Newton correction on the BF16 LSB.
inline void _calculate_reciprocal_fast_8b_3c_(const int iterations) {
#ifdef DISABLE_SFPLOADMACRO
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x8000);

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPMAD(p_sfpu::LCONST_0, p_sfpu::LCONST_0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LCONST_neg1, p_sfpu::LREG1, 0);
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG1, p_sfpu::LREG1, 5);
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
#else
    constexpr int y = p_sfpu::LREG0;
    constexpr int x = p_sfpu::LREG1;

    // Macro template 0 uses SFPMAD_MOD1_INDIRECT_VD, so LREG7 selects where
    // the source x copy lands.
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, x);

    // Pseudocode for the BF16 correction:
    //
    // y = load()
    // x = y
    // y = arecip(y)
    // y[15:0] = 0x8000
    // e = x * y - 1
    // t = e >> 16
    // y += t          # integer add, not FP32 add
    // store(y)
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TT_SFPLOADMACRO(
            /*lreg_ind*/ (0 << 2) | y,
            /*instr_mod0*/ InstrModLoadStore::DEFAULT,
            /*sfpu_addr_mode*/ ADDR_MOD_7,
            /*dest_reg_addr*/ 0);
        // Macro 0 schedules y = arecip(y) and x = y for the next SFPU issue.
        // Wait before writing y's low 16 bits directly.
        TTI_SFPNOP;
        // Keep the patch and correction in LReg space; macro store/reload
        // scheduling can read a just-written Dst block too soon on Blackhole.
        TTI_SFPLOADI(
            /*lreg_ind*/ y,
            /*instr_mod0*/ sfpi::SFPLOADI_MOD0_LOWER,
            /*imm16*/ 0x8000);
        TTI_SFPMAD(
            /*lreg_src_a*/ x,
            /*lreg_src_b*/ y,
            /*lreg_src_c*/ p_sfpu::LCONST_neg1,
            /*lreg_dest*/ x,
            /*instr_mod1*/ 0);
        TTI_SFPSHFT(
            /*imm12_math*/ (-16) & 0xFFF,
            /*lreg_c*/ x,
            /*lreg_dest*/ x,
            /*instr_mod1*/ 5);
        TTI_SFPIADD(
            /*imm12_math*/ 0,
            /*lreg_c*/ x,
            /*lreg_dest*/ y,
            /*instr_mod1*/ sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSTORE(
            /*lreg_ind*/ y,
            /*instr_mod0*/ InstrModLoadStore::DEFAULT,
            /*sfpu_addr_mode*/ ADDR_MOD_6,
            /*dest_reg_addr*/ 0);
    }

    TTI_SFPNOP;
#endif
}

// FP32 reciprocal, with throughput of 5c/32.
inline void _calculate_reciprocal_fast_24b_5c_(const int iterations) {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 1);  // SFPMAD_MOD1_NEGATE_VA
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
#else
    // Pseudocode:
    //
    // y = arecip(x)
    // e = 1 - x*y
    // t = e * e + e
    // t2 = t * e + e    # e**3 + e**2 + e
    // t2 = min(t2, 1.0) # replace NaN with 1.0
    // y = t2 * y + y    # y = y * (e**3 + e**2 + e + 1)
    //                   # if y = ±0 or ±inf, then y = y+y
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    //   | Load | Simple                 | MAD                     | Store   |
    // - | -----| ---------------------- | ----------------------- |-------- |
    // 0 | [y]  |                        |                         |         |
    // 1 |      | [y] = arecip(y)        |                         |         |
    // 2 | [e]  |                        |                         |         |
    // 3 |      | [e] L16 = arecip(e)    | e = mad(-e, y, 1.0)     |         |
    // 4 |      |                        |                         |         |
    // 0 |      |                        | [e] = mad(e, e, e)      | [e]     |
    // 1 | [t2] |                        |                         |         |
    // 2 |      |                        | [t2] = mad(t2, e, t2)   | [y] L16 |
    // 3 |      |                        |                         |         |
    // 4 | [z]  | [t2] = swap(t2, 1.0)   |                         |         |
    // 0 |      |                        |                         |         |
    // 1 |      |                        | [z] = mad(t2, z, z)     |         |
    // 2 |      |                        |                         |         |
    // 3 |      |                        |                         | [z]     |

    lltt::replay(0, 4);
    TTI_SFPLOAD(7, 0, ADDR_MOD_6, 0);

#pragma GCC unroll 7
    for (int d = 0; d < iterations - 1; d++) {
        lltt::replay(0, 5);
    }

    TTI_SFPNOP;
    lltt::replay(1, 1);
    TTI_SFPNOP;
    lltt::replay(3, 2);

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

// ~7b precision; 1c/element
inline void _init_reciprocal_fast_7b_() {
#ifndef DISABLE_SFPLOADMACRO
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t | Load | Simple                | Store   |
    // - | ---- | --------------------- | ------- |
    // 0 | [x]  |                       |         |
    // 1 |      | [x] L16 = arecip([x]) |         |
    // 2 |      |                       | [x] L16 |

    TTI_SFPARECIP(0, 0, 12, sfpi::SFPARECIP_MOD1_RECIP);

    constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
    constexpr std::uint32_t mad_bits = 0;
    constexpr std::uint32_t round_bits = 0;
    constexpr std::uint32_t store_bits = 0x00 | 0x40 | (1 << 3) | 3;

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);

    TTI_SFPCONFIG(0, 4, 0);

    // Misc: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for macro 0.
    TTI_SFPCONFIG(0x110, 8, 1);
#endif
}

inline void _init_reciprocal_fast_8b_3c_() {
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]
    TTI_SFPARECIP(
        /*imm12_math*/ 0,
        /*lreg_c*/ 0,
        /*lreg_dest*/ 12,
        /*instr_mod1*/ sfpi::SFPARECIP_MOD1_RECIP);

    // InstructionTemplate[1]
    TTI_SFPMAD(
        /*lreg_src_a*/ p_sfpu::LCONST_0,
        /*lreg_src_b*/ p_sfpu::LCONST_0,
        /*lreg_src_c*/ 0,
        /*lreg_dest*/ 13,
        /*instr_mod1*/ 8);  // SFPMAD_MOD1_INDIRECT_VD

    // Macro 0: [y]
    // Loads y, schedules y = arecip(y), and copies y to L[LREG7].
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0x00 | 0x00 | (0 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: 0, unused
    //   UsesLoadMod0ForStore: {0,0,0,0},
    //   UnitDelayKind: Simple + MAD use WaitForElapsedInstructions
    // }
    TTI_SFPCONFIG(0x300, 8, 1);
#endif
}

inline void _init_reciprocal_fast_24b_5c_() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int e = p_sfpu::LREG0;
    constexpr int t2 = p_sfpu::LREG1;
    constexpr int z = p_sfpu::LREG2;
    constexpr int y = p_sfpu::LREG3;

    // InstructionTemplate[0]
    TTI_SFPARECIP(0, 0, 12, sfpi::SFPARECIP_MOD1_RECIP);

    // InstructionTemplate[1]
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, 0, 13, 0);

    // InstructionTemplate[2]
    // SFPMAD(VA=t2, VB=0 or VD, VC=VD or z)
    TTI_SFPMAD(t2, p_sfpu::LREG0, z, 14, 0);

    // InstructionTemplate[3]
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, 15, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // Macro 0: [y]
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (6 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4, 0);
    }

    // Macro 1: [e]
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0x00 | 0x00 | (2 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2: [t2]
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (2 << 3) | (4 + 3);
        constexpr std::uint32_t mad_bits = 0x00 | 0x00 | (0 << 3) | (4 + 2);

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);
    }

    // Macro 3: [z]
    {
        // Keep the corrected result in z. Macro 0 still needs L16 for its delayed
        // store of the next vector's approximate reciprocal. Even with instruction-
        // counted delays, an issued MAD completes during scalar stalls (e.g. gcov),
        // so writing its result to L16 can clobber that value before macro 0 stores it.
        // The next load of z follows this macro's store, so z has no such overlap.
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0x80 | 0x00 | (1 << 3) | (4 + 2);
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for all macros.
    TTI_SFPCONFIG(0xff0, 8, 1);

    constexpr std::uint32_t prev_offset = -2 & 0x3ff;
    constexpr std::uint32_t offset = 0;

    load_replay_buf(0, 6, [e, t2, z, y, offset, prev_offset] {
        TTI_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));
        TTI_SFPLOADMACRO((2 << 2) | (t2 & 3), 0, ADDR_MOD_7, prev_offset | (t2 >> 2));
        TTI_SFPLOADMACRO((1 << 2) | (e & 3), 0, ADDR_MOD_7, offset | (e >> 2));
        TTI_SFPMAD(p_sfpu::LREG0, y, p_sfpu::LCONST_1, 0, 1);  // SFPMAD_MOD1_NEGATE_VA
        TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, ADDR_MOD_6, prev_offset | (z >> 2));
        TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, ADDR_MOD_7, prev_offset | (z >> 2));
    });
#endif
}

template <bool APPROXIMATE = false, bool save_reg = true /* Unused. Enough registers available. */>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in) {
    return sfpu_reciprocal_iter<APPROXIMATE ? 0 : 2>(in);
}

template <bool APPROXIMATE = false>
sfpi_inline void sfpu_reciprocal_init() {
    if constexpr (!APPROXIMATE) {
        sfpi::vConstFloatPrgm0 = 2.0f;
    }
}

// Fast bf16 reciprocal for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 327.1 cycles/tile vs 1202.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline). NOTE: that
// baseline is recip_tile's header default, legacy_compat = true (_calculate_reciprocal_compat_); this kernel is
// wired into the legacy_compat = false / !APPROX / bf16 slot in place of _calculate_reciprocal_fast_8b_3c_, whose
// cycles/tile were not measured by the bench -- compare against it when validating on hardware.
// SFPU state programmed by the init: programmable constant vConstFloatPrgm1 (LREG13) = 0.5f (LREG12 keeps
// sfpu_reciprocal_init's 2.0f, see recip_init); SFPLOADMACRO InstructionTemplate[0..3] (T0 0.5*x MAD, T1 CAST,
// T2 MULI 2^75, T3 SETSGN), LoadMacroConfig.Sequence[0..3] (macros 0..3) + Misc; all lanes enabled (SFPENCC).
// No replay slots. ADDR_MOD_6 = dest incr 2 (programmed by recip_init) and ADDR_MOD_7. bf16 DEST only (the
// SETSGN/CAST bit tricks assume bf16 bit patterns) -- gated on !is_fp32_dest_acc_en; the existing paths stay
// as the fallback for every other configuration and for DISABLE_SFPLOADMACRO builds.
//
// Region-free algorithm (validated exhaustively against documented models):
//   C = x*0.5 + 0                   ; |C| == 0  <=>  exp(x) <= 1
//   B = arecipU(C) * 0.5*(1+2^-7)   ; unsigned main path (+inf on fixup lanes)
//   D = arecipU(cast_sm32(bits(x))) * 2^74 * 2^75
//                                   ; unsigned fixup path; == |1/x| on fixup
//                                     lanes, provably >= |1/x| on normal lanes
//   B = min(B, D)                   ; SFPSWAP vec min/max -- no lane predication
//   out = sign(x) | B               ; SFPSETSGN into the sign-carrying reg
// Max ULP 1 exhaustively. No CC anywhere.
//
// Two vectors (even/odd) per 17-issue block, software-pipelined across blocks
// (each block loads the next block's C). SFPLOADMACRO fuses the in-place MAD,
// the CAST, the final D*2^75, the even-vector SETSGN and both stores. Macro
// delays count issued instructions (Misc=0xFF0) so the schedule is issue-
// indexed and stall-tolerant. Constraints honored: macro loads only target
// L0..L3 (VDHi doubles as the address LSB); no macro op fires during either
// execution cycle of an SFPSWAP.
//   T = L0 (sign source + store data), D_e = L2, B_e = L4, B_o = L6,
//   C_e/C_o/D_o rotate over {L1, L3} (D_o reuses the dead C_e register).
// Macros: 0 = load C + {MAD C=0.5*C @d2}
//         1 = load D + {CAST D @d0, MULI D*=2^75 @d6}
//         2 = load T + {SETSGN T=sgn(T)|mag(B_e=L4) @d0, STORE(T) @d1}
//         3 = load T + {STORE(T) @d1}
#ifndef DISABLE_SFPLOADMACRO
inline void _init_reciprocal_bf16_fast_() {
    // recip_init has already run the common prologue (SFPU config reg, ADDR_MOD_7, ADDR_MOD_6 = dest incr 2,
    // counter reset) and sfpu_reciprocal_init<false>() (vConstFloatPrgm0 = 2.0f). The 0.5f constant therefore
    // lives in vConstFloatPrgm1 (LREG13) rather than vConstFloatPrgm0 (LREG12) as in the original kernel, so
    // callers that follow recip_init<false, false, false>() with sfpu_reciprocal_iter (e.g. the moe_gpt SwiGLU
    // kernel) keep the 2.0f Newton-Raphson constant they rely on. T0 below reads VA = LREG13 accordingly.
    sfpi::vConstFloatPrgm1 = 0.5f;  // LREG13
    TTI_SFPENCC(3, 0, 0, 10);       // all lanes enabled

    // Instruction templates (T0..T2 via backdoor write: VD in 12..14 captures).
    TTI_SFPMAD(13, 0, 9, 12, 0);  // T0: VD = 0.5*VB + 0    (VB,VD <- macro load reg; VA = LREG13 = 0.5f)
    TTI_SFPCAST(0, 13, 0);        // T1: cast               (VC,VD <- macro load reg)
    TTI_SFPMULI(0x6500, 14, 0);   // T2: *= 2^75            (VC=VD <- macro load reg)
    {
        // T3: VD = sign(VB)|mag(L4); VB,VD <- macro load reg. Written via
        // SFPCONFIG because SETSGN's write gate does not backdoor-capture.
        constexpr std::uint32_t setsgn_word = TT_OP_SFPSETSGN(0, 4, 0, 0);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, setsgn_word & 0xFFFF);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, setsgn_word >> 16);
        TTI_SFPCONFIG(0, 3, 0);
    }
    // Sequence bytes: [store]<<24 | [round]<<16 | [mad]<<8 | [simple];
    // byte = 0x80(VB<-load) | 0x40(VD=L16) | delay<<3 | slot(3=builtin store, 4+i=template i)
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x9400);  // macro0: mad = T0 d2
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPCONFIG(0, 4 + 0, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x3605);  // macro1: simple = T1 d0; mad = T2 d6
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPCONFIG(0, 4 + 1, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x0087);  // macro2: simple = T3 d0; store = builtin d1
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0B00);
    TTI_SFPCONFIG(0, 4 + 2, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);  // macro3: store = builtin d1
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0B00);
    TTI_SFPCONFIG(0, 4 + 3, 0);
    // Misc: UnitDelayKind = 0xF (instruction-counted), UsesLoadMod0ForStore = 0xF.
    TTI_SFPCONFIG(0xFF0, 8, 1);
}

// One even/odd pair. CE/CO = this block's C regs; D_o reuses CE's register.
// The b14 load fetches the NEXT block's C_e into CO's register (dead by then).
#define RECIP_FAST_PAIR(CE, CO)                                                                  \
    TTI_SFPLOADMACRO((1 << 2) | 2, 0, ADDR_MOD_6, 0);      /* b1  D_e=x_e {CAST@2,*2^75@8};  */ \
                                                           /*     ctr += 2                   */ \
    TTI_SFPLOADMACRO((0 << 2) | (CO), 0, ADDR_MOD_7, 0);   /* b2  C_o=x_o {MAD@5}            */ \
    TTI_SFPARECIP(11, 2, 2, 1);                            /* b3  D_e = arecipU(D_e)         */ \
    TTI_SFPMULI(0x6480, 2, 0);                             /* b4  D_e *= 2^74                */ \
    TTI_SFPARECIP(11, (CE), 4, 1);                         /* b5  B_e = arecipU(C_e)         */ \
    TTI_SFPLOADMACRO((1 << 2) | (CE), 0, ADDR_MOD_7, 0);   /* b6  D_o=x_o {CAST@7,*2^75@13}  */ \
    TTI_SFPMULI(0x3F01, 4, 0);                             /* b7  B_e *= 0.5*(1+2^-7)        */ \
    TTI_SFPARECIP(11, (CE), (CE), 1);                      /* b8  D_o = arecipU(D_o)         */ \
    TTI_SFPMULI(0x6480, (CE), 0);                          /* b9  D_o *= 2^74                */ \
    TTI_SFPARECIP(11, (CO), 6, 1);                         /* b10 B_o = arecipU(C_o)         */ \
    TTI_SFPSWAP(0, 2, 4, 1);                               /* b11 B_e = min(B_e, D_e)        */ \
    TTI_SFPMULI(0x3F01, 6, 0);                             /* b12 B_o *= 0.5*(1+2^-7)        */ \
    TTI_SFPLOADMACRO((2 << 2) | 0, 0, ADDR_MOD_7, 0x3FE);  /* b13 T=x_e@-2 {SETSGN@14,ST@15} */ \
    TTI_SFPLOADMACRO((0 << 2) | (CO), 0, ADDR_MOD_7, 2);   /* b14 next C_e (CO reg) {MAD@17} */ \
    TTI_SFPSWAP(0, (CE), 6, 1);                            /* b15 B_o = min(B_o, D_o)        */ \
    TTI_SFPLOADMACRO((3 << 2) | 0, 0, ADDR_MOD_6, 0);      /* b16 T=x_o {ST@18}; ctr += 2    */ \
    TTI_SFPSETSGN(0, 6, 0, 0);                             /* b17 T = sign(T)|mag(B_o)       */

// One face (8 dst vectors) as 4 even/odd blocks.
inline void _calculate_reciprocal_bf16_fast_() {
    TTI_SFPLOADMACRO((0 << 2) | 1, 0, ADDR_MOD_7, 0);  // prologue: C_e(L1) = x {MAD@+3}
    RECIP_FAST_PAIR(1, 3)  // block 0: C_e=L1 C_o=L3 D_o=L1; loads block1's C_e into L3
    RECIP_FAST_PAIR(3, 1)  // block 1
    RECIP_FAST_PAIR(1, 3)  // block 2
    RECIP_FAST_PAIR(3, 1)  // block 3 (its next-C load reads past the face; harmless)
    TTI_SFPNOP;            // flush the final pending store (fires on this issue)
}

#undef RECIP_FAST_PAIR
#endif  // DISABLE_SFPLOADMACRO

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8, bool legacy_compat = false>
inline void calculate_reciprocal() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!legacy_compat && !APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_reciprocal_bf16_fast_();
        return;
    }
    if constexpr (
        !legacy_compat && !APPROXIMATION_MODE && !is_fp32_dest_acc_en &&
        !(!legacy_compat && !APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8 (tt-llk tests): recip_init<false, false> cannot see ITERATIONS and has programmed the
        // fast kernel's SFPLOADMACRO InstructionTemplate[0..3] / Sequence[0..3] / Misc in place of the 8b_3c macro
        // state _calculate_reciprocal_fast_8b_3c_ depends on; re-program it before falling back. The fast init
        // leaves vConstFloatPrgm0 = 2.0f (sfpu_reciprocal_init) untouched and 8b_3c does not read vConstFloatPrgm1.
        _init_reciprocal_fast_8b_3c_();
    }
#endif
    if constexpr (legacy_compat) {
        _calculate_reciprocal_compat_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(ITERATIONS);
    } else if constexpr (APPROXIMATION_MODE) {
        _calculate_reciprocal_fast_7b_(ITERATIONS);
    } else if constexpr (is_fp32_dest_acc_en) {
        _calculate_reciprocal_fast_24b_5c_(ITERATIONS);
    } else {
        _calculate_reciprocal_fast_8b_3c_(ITERATIONS);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
void recip_init() {
    // Common SFPU init inlined (SFPU config register + ADDR_MOD_7 + reciprocal's ADDR_MOD_6 + counter
    // reset), then the op-specific reciprocal setup below -- one self-contained init, matching exp_init.
    // SDPA runs reciprocal in its softmax after matmul/exp, so the general SFPU state is re-established
    // here, not just reset. Reciprocal uses ADDR_MOD_6 (dest incr 2) on Blackhole.
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!legacy_compat) {
        sfpu_reciprocal_init<false>();  // set vConstFloatPrgm0 for sfpu_reciprocal_iter
        if constexpr (APPROXIMATION_MODE) {
            _init_reciprocal_fast_7b_();
        } else if constexpr (is_fp32_dest_acc_en) {
            _init_reciprocal_fast_24b_5c_();
        } else {
#ifndef DISABLE_SFPLOADMACRO
            // Fast bf16 kernel (see _init_reciprocal_bf16_fast_) replaces the 8b_3c macro path on this combination.
            _init_reciprocal_bf16_fast_();
#else
            _init_reciprocal_fast_8b_3c_();
#endif
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
