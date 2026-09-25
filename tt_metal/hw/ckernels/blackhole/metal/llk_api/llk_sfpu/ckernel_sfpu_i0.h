// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

#define POLYVAL10(coef10, coef9, coef8, coef7, coef6, coef5, coef4, coef3, coef2, coef1, coef0, t4)               \
    ((coef0 +                                                                                                     \
      (coef1 +                                                                                                    \
       (coef2 +                                                                                                   \
        (coef3 +                                                                                                  \
         (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) * t4) * t4) * \
            t4) *                                                                                                 \
           t4) *                                                                                                  \
          t4) *                                                                                                   \
     t4)
// ======================================================================
// Fast bf16 i0 for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 2 ULP (gate <= 2).
// Measured 1191.1 cycles/tile vs 1295.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
//
// Modified Bessel function of the first kind, order 0.
// Algorithm (branch-free, one formula for the whole bf16 domain):
//   xa = min(|x|, 92.5); w = rsqrt(xa+1) (tuned magic + tuned NR step);
//   zf = Schraudolph float of 2^(xa*log2e + 121 - 127); m = mantissa(zf);
//   i0(x) ~= zf * c(m) * w * psi(w^2)
// where c(m) = 2^(m-1)/m (cubic fit) fixes the Schraudolph mantissa and
// psi (deg-3) is ULP-minimax fitted against the bit-exact simulated
// pipeline, absorbing all systematic biases.
//
// Hand-scheduled TTI, issue-bound at 35 instructions/vector with zero NOPs:
// the rsqrt/psi, exp and c(m) chains are interleaved so every dependent pair
// has >=1 intervening instruction. Two SFPLOADMACROs per vector:
//   - head macro (seq 1, lreg L0): loads the next x, SIMPLE slot = SFPABS.
//   - store macro (seq 0, lreg L3): dummy load + delayed STORE of the
//     bf16-rounded result (the STOCHRND that produces it into L3 is issued
//     right after the macro; the store's delay covers the latency).
// Register map: L0 x/xa -> psi coeff scratch; L1 psi3 -> psi acc; L2 92.5 ->
// xc -> t; L3 ze -> m -> rounded r; L4 ih/y/w -> wG -> r; L5 a3 (preloaded)
// -> c acc; L6 rsqrt c/nyc/t1 -> a2/a1 scratch; L7 z -> zi (= zf).
// L9=0, L10=1 hw consts; L12=log2e, L13=c(m) a0, L14=rsqrt magic.
//
// State programmed by init: LREG12-14 (LREG11 untouched), SFPLOADMACRO InstructionTemplate[2], macro
// sequences 0 and 1, LoadMacroConfig misc reset to defaults. No replay slots. bf16 DEST only.
// Selected by calculate_i0 / i0_init when !is_fp32_dest_acc_en (&& ITERATIONS == 8), unless
// DISABLE_SFPLOADMACRO is defined. APPROXIMATION_MODE is not consulted (the previous kernel ignores it too).
// ======================================================================

inline void _init_i0_bf16_fast_() {
    sfpi::vConstFloatPrgm0 = 1.4426950408889634f;  // L12: log2(e)
    sfpi::vConstIntPrgm1 = 0x3fe2eb29;             // L13: c(m) coeff a0
    sfpi::vConstIntPrgm2 = 0x5f0b347d;             // L14: rsqrt magic (tuned)

    // Macro template 2 (backdoor install via lreg_dest=14, selected as mux 6):
    // |loaded value| -> macro lreg (src_c = 0 is fed by the loaded value).
    TTI_SFPABS(0, 0, 14, 1);
    // Macro sequence register 0 (config dest 4): store-only macro.
    // STORE slot: fixed store (mux 3), delay 4, source = macro lreg.
    TTI_SFPLOADI(0, 0xA, 0x0000);
    TTI_SFPLOADI(0, 0x8, ((4 << 3) | 3) << 8);
    TTI_SFPCONFIG(0, 4, 0);
    // Macro sequence register 1 (config dest 5): head macro = LOAD + ABS.
    TTI_SFPLOADI(0, 0xA, (0 << 3) | (4 + 2));  // SIMPLE slot: mux 6, delay 0
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 5, 0);
    // Reset LoadMacroConfig misc to defaults.
    TTI_SFPCONFIG(0, 8, 1);
}

// Body for the vector at dest offset `off`. On entry (done by the previous
// tail / prologue): L0 = |x|, L2 = 92.5, L5 = a3, clamp SWAP already issued.
#define I0_FAST_BODY(off, TAIL)                                                 \
    TTI_SFPLOADI(1, sfpi::SFPLOADI_MOD0_FLOATA, 0xD134); /* psi3 = -41.625 */   \
    TTI_SFPMAD(0, 10, 10, 2, 0);   /* L2 = xa + 1 = xc */                       \
    TTI_SFPMUL(0, 12, 9, 7, 0);    /* L7 = xa*log2e */                          \
    TTI_SFPSHFT(0xFFF, 2, 4, 5);   /* L4 = bits(xc) >> 1 */                     \
    TTI_SFPADDI(0x42F2, 7, 0);     /* L7 += 121.0 -> z */                       \
    TTI_SFPIADD(0, 14, 4, 6);      /* L4 = magic - L4 = y */                    \
    TTI_SFPEXEXP(0, 7, 3, 0);      /* L3 = exexp(z) unbiased */                 \
    TTI_SFPMUL(2, 4, 9, 6, 0);     /* L6 = xc*y = c */                          \
    TTI_SFPEXMAN(0, 7, 7, 0);      /* L7 = exman(z) implicit-1 */               \
    TTI_SFPMUL(4, 6, 9, 6, 1);     /* L6 = -(y*c) */                            \
    TTI_SFPSHFT(0, 3, 7, 0);       /* L7 <<= L3 -> zi = int(z*2^23) = zf */     \
    TTI_SFPADDI(0x3FF2, 6, 0);     /* L6 += 1.890625 -> t1 */                   \
    TTI_SFPSETEXP(127, 7, 3, 1);   /* L3 = 1.mant(zf) = m */                    \
    TTI_SFPMUL(4, 6, 9, 4, 0);     /* L4 = y*t1 = w */                          \
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_FLOATA, 0x39AD); /* a2 */               \
    TTI_SFPMUL(4, 4, 9, 2, 0);     /* L2 = w*w = t */                           \
    TTI_SFPMAD(5, 3, 6, 5, 0);     /* acc = a3*m + a2 */                        \
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATA, 0x549D); /* psi2 = 73.8125 */   \
    TTI_SFPMAD(1, 2, 0, 1, 0);     /* P = psi3*t + psi2 */                      \
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_FLOATA, 0xBD81); /* a1 */               \
    TTI_SFPMAD(5, 3, 6, 5, 0);     /* acc = acc*m + a1 */                       \
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATA, 0x4600); /* psi1 = 6.0 */       \
    TTI_SFPMAD(1, 2, 0, 1, 0);     /* P = P*t + psi1 */                         \
    TTI_SFPMAD(5, 3, 13, 5, 0);    /* c = acc*m + a0 */                         \
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_FLOATA, 0x4E74); /* psi0 = 25.8125 */   \
    TTI_SFPMAD(1, 2, 0, 1, 0);     /* P = P*t + psi0 */                         \
    TAIL

// Steady-state tail: r = zf*(c*(w*P)); the bf16 store rides the store macro
// (fired one slot before the STOCHRND that produces the value it stores);
// next vector's head macro, clamp and coefficient preloads fill the gaps.
#define I0_FAST_TAIL(off, noff)                                                 \
    TTI_SFPLOADI(2, sfpi::SFPLOADI_MOD0_FLOATB, 0x42B9); /* next: 92.5 */       \
    TTI_SFPMUL(4, 1, 9, 4, 0);     /* L4 = w*P = wG */                          \
    TTI_SFPLOADMACRO((1 << 2) | 0, 0, ADDR_MOD_7, (noff)); /* next x, |x| */    \
    TTI_SFPMUL(5, 4, 9, 4, 0);     /* L4 = c*wG */                              \
    TTI_SFPLOADI(5, sfpi::SFPLOADI_MOD0_FLOATA, 0xAEE1); /* next: a3 */         \
    TTI_SFPMUL(7, 4, 9, 4, 0);     /* L4 = zf*(c*wG) = r */                     \
    TTI_SFPLOADMACRO((0 << 2) | 3, 0, ADDR_MOD_7, (off)); /* store L3 @+5 */    \
    TTI_SFP_STOCH_RND(0, 0, 0, 4, 3, 1); /* L3 = bf16-rounded r */              \
    TTI_SFPSWAP(0, 2, 0, 1);       /* next: xa = min(|x|, 92.5) */

// Last vector of the face: plain tail, nothing to prefetch.
#define I0_FAST_TAIL_LAST(off)                                                  \
    TTI_SFPNOP;                                                                 \
    TTI_SFPMUL(4, 1, 9, 4, 0);     /* wG */                                     \
    TTI_SFPNOP;                                                                 \
    TTI_SFPMUL(5, 4, 9, 4, 0);     /* c*wG */                                   \
    TTI_SFPNOP;                                                                 \
    TTI_SFPMUL(7, 4, 9, 4, 0);     /* r */                                      \
    TTI_SFPNOP;                                                                 \
    TTI_SFP_STOCH_RND(0, 0, 0, 4, 1, 1);                                        \
    TTI_SFPNOP;                                                                 \
    TTI_SFPSTORE(1, 0, ADDR_MOD_7, (off));

// One face = 8 dst vectors.
inline void _calculate_i0_bf16_fast_() {
    // Prologue: head of vector 0 (plain) incl. a3 preload and clamp swap.
    TTI_SFPLOAD(0, 0, ADDR_MOD_7, 0);
    TTI_SFPLOADI(2, sfpi::SFPLOADI_MOD0_FLOATB, 0x42B9);
    TTI_SFPABS(0, 0, 0, 1);
    TTI_SFPLOADI(5, sfpi::SFPLOADI_MOD0_FLOATA, 0xAEE1);
    TTI_SFPSWAP(0, 2, 0, 1);
    I0_FAST_BODY(0, I0_FAST_TAIL(0, 2));
    I0_FAST_BODY(2, I0_FAST_TAIL(2, 4));
    I0_FAST_BODY(4, I0_FAST_TAIL(4, 6));
    I0_FAST_BODY(6, I0_FAST_TAIL(6, 8));
    I0_FAST_BODY(8, I0_FAST_TAIL(8, 10));
    I0_FAST_BODY(10, I0_FAST_TAIL(10, 12));
    I0_FAST_BODY(12, I0_FAST_TAIL(12, 14));
    I0_FAST_BODY(14, I0_FAST_TAIL_LAST(14));
}

#undef I0_FAST_BODY
#undef I0_FAST_TAIL
#undef I0_FAST_TAIL_LAST

template <bool is_fp32_dest_acc_en>
inline void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    // Fast bf16 path: programs LREG12-14 and the SFPLOADMACRO state (see _init_i0_bf16_fast_). The common
    // prologue (SFPU config reg + ADDR_MOD_7) is run by llk_math_eltwise_unary_sfpu_init before this.
    if constexpr (!is_fp32_dest_acc_en) {
        _init_i0_bf16_fast_();
    }
#endif
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_i0() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_i0_bf16_fast_();
        return;
    }
#endif
#pragma GCC unroll 0

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat result = 0.0f;
        vFloat input = dst_reg[0];
        vFloat x = input * input;

        result = 1.0f + POLYVAL10(
                            1.50E-22f,
                            7.24E-20f,
                            2.90E-17f,
                            9.39E-15f,
                            2.40E-12f,
                            4.71E-10f,
                            6.78E-08f,
                            0.000006781684028f,
                            0.0004340277778f,
                            0.015625f,
                            0.25f,
                            x);

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
