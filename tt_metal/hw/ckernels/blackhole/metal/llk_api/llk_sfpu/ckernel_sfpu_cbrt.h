// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// This is a modified version of "Fast Calculation of Cube and Inverse Cube
// Roots Using a Magic Constant and Its Implementation on Microcontrollers" by
// Moroz et al. <https://doi.org/10.3390/en14041058>

// ======================================================================
// Fast bf16 cbrt for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 622.1 cycles/tile vs 771.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
//
// cbrt(x) for bf16, raw TTI, 2-vector pipeline + LOADMACRO.
// Algorithm:
//   x = |a|; e = biased exponent field of a
//   if (e == 0): x = float(bits(x)) * 2^-125    // rebuild denormal as x*2^24
//   i = float(bits(x)); f = i*N + M; y = bits(f)<<8 reinterpreted as float
//   w = y*y; d = x*w; c = d*y; u = c*B + 1; r = d*u
//   if (e == 0): r *= 2^-8
//   r = copysign(r, a); store r  (truncating bf16 store; constants absorb it)
//
// Per pair of vectors: A uses L0-L3, B uses L4-L7. The tail (copysign +
// store) rides SFPLOADMACRO-scheduled Simple/Store slots hung off the sign
// reload; B's reload targets L2 (macro loads must target L0-L3 because the
// VD high bit aliases the address LSB).
//
// State programmed by init: LREG12 (N), LREG13 (M), LREG14 (B) -- these REPLACE the vConstFloatPrgm0..2
// values programmed by cube_root_init for the generic path; SFPLOADMACRO InstructionTemplate[0..2],
// macro sequences 0..2, LoadMacroConfig misc = 0. No replay slots. bf16 DEST only.
// Selected by calculate_cube_root / cube_root_init when !APPROXIMATION_MODE && !is_fp32_dest_acc_en
// (&& ITERATIONS == 8), unless DISABLE_SFPLOADMACRO is defined.
// ======================================================================

inline void _init_cbrt_bf16_fast_() {
    sfpi::vConstFloatPrgm0 = -0x1.555556p-10f;  // N  (CREG 12)
    sfpi::vConstFloatPrgm1 = 0x1.a9a2bap+23f;   // M  (CREG 13)
    sfpi::vConstFloatPrgm2 = -0x1.7a204ep-3f;   // B  (CREG 14)

    // Instruction templates for SFPLOADMACRO (programmed by issuing with
    // VD = 12+t): T0/T1 = copysign for vector A/B: VD = {VB.sign, VC.mag},
    // VB and VD are overridden at fire time with the macro's load register.
    TTI_SFPSETSGN(0, 3, 12, 0);                     // T0: magnitude from L3 (rA)
    TTI_SFPSETSGN(0, 7, 13, 0);                     // T1: magnitude from L7 (rB)
    TTI_SFPABS(0, 0, 14, sfpi::SFPABS_MOD1_FLOAT);  // T2: abs (VC overridden)

    // Macro 0 (tail A): Simple = T0 @delay0 (VB-override), Store @delay2.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x0084);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x1300);
    TTI_SFPCONFIG(0, 4 + 0, 0);
    // Macro 1 (tail B): Simple = T1 @delay0, Store @delay2.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x0085);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x1300);
    TTI_SFPCONFIG(0, 4 + 1, 0);
    // Macro 2 (fired by vector A's main load): Simple = T2 (abs) @delay0.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, 0x0006);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPCONFIG(0, 4 + 2, 0);
    // Misc: StoreMod0 = DEFAULT, all delays cycle-counted.
    TTI_SFPCONFIG(0x000, 8, 1);
}

#define CBRT_FAST_PAIR(OFFA, OFFB)                                             \
    /* front */                                                                \
    TTI_SFPLOADMACRO(8, 0, ADDR_MOD_7, OFFA); /* macro2 VD=L0: load + abs */  \
    TTI_SFPLOAD(4, 0, ADDR_MOD_7, OFFB);                                       \
    TTI_SFPEXEXP(0, 4, 5, sfpi::SFPEXEXP_MOD1_NODEBIAS);                       \
    TTI_SFPEXEXP(0, 0, 1, sfpi::SFPEXEXP_MOD1_NODEBIAS);                       \
    TTI_SFPABS(0, 4, 4, sfpi::SFPABS_MOD1_FLOAT);                              \
    TTI_SFPSETCC(0, 1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);                       \
    TTI_SFPCAST(0, 0, 0);              /* denorm: L0 = float(bits(x)) */       \
    TTI_SFPMULI(0x0100, 0, 0);         /* denorm: L0 *= 2^-125 -> X   */       \
    TTI_SFPENCC(0x003, 0, 0, 10);                                              \
    TTI_SFPSETCC(0, 5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);                       \
    TTI_SFPCAST(4, 4, 0);                                                      \
    TTI_SFPMULI(0x0100, 4, 0);                                                 \
    TTI_SFPENCC(0x003, 0, 0, 10);                                              \
    TTI_SFPCAST(0, 2, 0);              /* i_A */                               \
    TTI_SFPCAST(4, 6, 0);              /* i_B */                               \
    /* mid: guess + correction, A/B interleaved */                             \
    TTI_SFPMAD(2, 12, 13, 2, 0);       /* f = i*N + M */                       \
    TTI_SFPMAD(6, 12, 13, 6, 0);                                               \
    TTI_SFPSHFT(8, 2, 2, 5);           /* y bits = f bits << 8 */              \
    TTI_SFPSHFT(8, 6, 6, 5);                                                   \
    TTI_SFPMAD(2, 2, 9, 3, 0);         /* w = y*y */                           \
    TTI_SFPMAD(6, 6, 9, 7, 0);                                                 \
    TTI_SFPMAD(0, 3, 9, 3, 0);         /* d = x*w */                           \
    TTI_SFPMAD(4, 7, 9, 7, 0);                                                 \
    TTI_SFPMAD(3, 2, 9, 2, 0);         /* c = d*y */                           \
    TTI_SFPMAD(7, 6, 9, 6, 0);                                                 \
    TTI_SFPMAD(2, 14, 10, 2, 0);       /* u = c*B + 1 */                       \
    TTI_SFPMAD(6, 14, 10, 6, 0);                                               \
    TTI_SFPMAD(3, 2, 9, 3, 0);         /* r = d*u */                           \
    TTI_SFPMAD(7, 6, 9, 7, 0);                                                 \
    /* tail: denormal 2^-8 fixup on |r|, then macro copysign + store */        \
    TTI_SFPSETCC(0, 1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);                       \
    TTI_SFPMULI(0x3B80, 3, 0);                                                 \
    TTI_SFPENCC(0x003, 0, 0, 10);                                              \
    TTI_SFPSETCC(0, 5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);                       \
    TTI_SFPMULI(0x3B80, 7, 0);                                                 \
    TTI_SFPENCC(0x003, 0, 0, 10);                                              \
    TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, OFFA); /* macro0 VD=L2: sgnA, stA */   \
    TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7, OFFB); /* macro1 VD=L1: sgnB, stB */

// One face = 8 dst vectors (four pairs).
inline void _calculate_cbrt_bf16_fast_() {
    CBRT_FAST_PAIR(0, 2)
    CBRT_FAST_PAIR(4, 6)
    CBRT_FAST_PAIR(8, 10)
    CBRT_FAST_PAIR(12, 14)
}

#undef CBRT_FAST_PAIR

// Refinement constants read by calculate_cube_root's generic (sfpi) body from vConstFloatPrgm0..2. cube_root_init
// programs these, then -- on the fast bf16 gate -- the fast kernel's constants on top (same LREGs); the
// ITERATIONS != 8 fallback in calculate_cube_root re-seeds them.
inline void _init_cbrt_body_constants_() {
    sfpi::vConstFloatPrgm0 = 0x1.c09806p0f;
    sfpi::vConstFloatPrgm1 = -0x1.403e6cp0f;
    sfpi::vConstFloatPrgm2 = 0x1.04cdb2p-1f;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_cbrt_bf16_fast_();
        return;
    }
    if constexpr (
        (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) &&
        !(!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8: cube_root_init<false, false> cannot see ITERATIONS and has programmed the fast
        // kernel's LREG12-14 over the refinement constants this path reads from vConstFloatPrgm0..2; re-seed.
        _init_cbrt_body_constants_();
    }
#endif
    sfpi::vFloat negative_third_256 = -0x1.555556p-10f;

    // Magic constant 0x548c2b4b / 256 + 2^23
    sfpi::vFloat magic = 1418472267.0f / 256.0f + 8388608.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat x = sfpi::abs(a);

        // Paper wants i = 0x548c2b4b - i/3.
        // Due to lack of integer division and lack of fp32 to u32 cast, we
        // compute this using two instructions: SFPMAD and SFPSHFT.
        //
        // First, we compute (0x548c2b4b - i/3) in fp32, but we also need to
        // add 2^23 to shift the result into the mantissa bits for extraction
        // as integer.  This only works if (0x548c2b4b - i/3)*k < 2^23, so we
        // divide everything by 2^8.
        //
        // f = (0x548c2b4b - i * 1.0/3.0) / 256.0 + 2^23
        //   = (0x548c2b4b/256.0 - i * 1.0/3.0/256.0) + 2^23

        sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(x), sfpi::RoundMode::Nearest);

        f = f * negative_third_256 + magic;

        // Now, left-shift by 8 to restore integer result.

        sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(f) << 8);

        if constexpr (is_fp32_dest_acc_en) {
            sfpi::vFloat c = (x * y) * (y * y);
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0);

            sfpi::vFloat d = x * (y * y);
            c = d * y + -1.0f;
            sfpi::vFloat negative_third = sfpi::addexp(negative_third_256, 8);
            sfpi::vFloat t = c * negative_third + 1.0f;
            d = sfpi::copysgn(d, a);
            y = d * (t * t);
        } else {
            sfpi::vFloat d = x * (y * y);
            sfpi::vFloat c = d * y;
            sfpi::vFloat t = c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0;
            d = sfpi::copysgn(d, a);
            y = d * (t * t);
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void cube_root_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    _init_cbrt_body_constants_();
#ifndef DISABLE_SFPLOADMACRO
    // Fast bf16 path: re-programs LREG12-14 (overriding the three values above) and the SFPLOADMACRO
    // state (see _init_cbrt_bf16_fast_). The common prologue (SFPU config reg + ADDR_MOD_7) is run by the
    // llk_math_eltwise_unary_sfpu_init callback overload before this function is called.
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        _init_cbrt_bf16_fast_();
    }
#endif
}

}  // namespace ckernel::sfpu
