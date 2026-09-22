// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The log1p(x) code is derived from code by Norbert Juffa.
 *
 * Copyright (c) 2015-2023, Norbert Juffa
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_log.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel {
namespace sfpu {

// Production constant setup (defined with the init below); declared early so the ITERATIONS != 8
// fallback re-seed inside the calculate function can name it.
template <bool is_fp32_dest_acc_en>
inline void _init_log1p_body_constants_();

// For inputs with u = 1 + a > 0, write u = 2^k * t with t chosen in [0.75, 1.5),
// so m = t - 1 lies in [-0.25, 0.5). Then
//   log1p(a) = log(u) = k * log(2) + log1p(m).
// This implementation carries k in exponent-bit units (k << 23), which makes the
// rescaling and final k * log(2) term cheaper on SFPU.
// Inputs with 1 + a < 0 fall through to the default NaN.
// The boundary case u == 0 (a == -1) is outside the 2^k * t derivation above;
// it still evaluates to -inf via the same bit-level reduction path.
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_fp32(sfpi::vFloat a) {
    sfpi::vFloat u = a + 1.0f;
    sfpi::vFloat r = std::numeric_limits<float>::quiet_NaN();

    v_if(u >= 0.0f) {
        sfpi::vFloat three_quarters = 0.75f;
        sfpi::vInt e = sfpi::as<sfpi::vInt>(three_quarters);
        sfpi::vFloat e_float;

        // Subtracting the encoding of 0.75 and then zeroing the mantissa isolates
        // the exponent-bit offset e = k << 23 for the unique k with
        // 2^(-k) * u in [0.75, 1.5).
        e = sfpi::as<sfpi::vInt>(u) - e;
        e = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));

        // Reinterpreting a - e applies the same 2^(-k) scaling to a.
        // Affine correction below reconstructs
        //   m <- 2^(-k) * a + (2^(-k) - 1) = 2^(-k) * (1 + a) - 1.
        sfpi::vFloat m = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - e);
        sfpi::vFloat neg_four = -4.0f;
        // Use s' = -4 * 2^(-k) instead of 4 * 2^(-k); see -0.25 for explanation.
        sfpi::vFloat s = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(neg_four) - e);

        // Use -0.25 (instead of 0.25) so we can reuse it in the bf16 Horner step later.
        sfpi::vFloat neg_quarter = -0.25f;
        sfpi::vFloat neg1 = -1.0f;
        // t = -s' / 4 - 1 = 2^(-k) - 1
        sfpi::vFloat t = __builtin_rvtt_sfpmad(neg_quarter.get(), s.get(), neg1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        sfpi::vMag abs_e = sfpi::abs(e);

        // Minimax approximations for log1p(m) on [-0.25, 0.5]. Both paths keep the
        // exact linear term m explicit and approximate only the nonlinear
        // correction m^2 * P(m); the fp32 path keeps more terms than the
        // bf16-rounded path. fp16 or bf16 constants are used where possible to
        // reduce instruction count.
        if constexpr (is_fp32_dest_acc_en) {
            // log1p(m) ~= m + m*m * (
            //   -0x1p-1 + m * (0x1.555572p-2 + m * (-0x1.00001ap-2 + m * (0x1.998p-3 +
            //   m * (-0x1.55p-3 + m * (0x1.274p-3 + m * (-0x1.0c4p-3 + m *
            //   (0x1.b84p-4 + m * (-0x1.92cp-5)))))))))

            m = m + t;
            r = -0x1.92cp-5f;
            r = r * m + 0x1.b84p-4f;
            r = r * m + -0x1.0c4p-3f;
            r = r * m + 0x1.274p-3f;
            r = r * m + -0x1.55p-3f;
            r = r * m + 0x1.998p-3f;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
            r = r * m + sfpi::vConstFloatPrgm1;
            s = m * m;
            r = r * m + sfpi::vConstFloatPrgm2;
            r = r * m + -0.5f;
        } else {
            // log1p(x) = x + x*x * (-0x1.008p-1 + x * (0x1.744p-2 + x * (-0x1p-2)))

            m = m + t;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
            r = neg_quarter * m + sfpi::vConstFloatPrgm1;
            s = m * m;
            r = r * m + sfpi::vConstFloatPrgm2;
        }
        // convert<vFloat> returns |e| as a real number in exponent-bit units;
        // restore sign and multiply by log(2) * 2^(-23) to recover k * log(2).
        e_float = sfpi::copysgn(e_float, sfpi::as<sfpi::vFloat>(e));
        r = r * s + m;
        sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
        r = e_float * sfpi::vConstFloatPrgm0 + r;

        // since u>=0, safely checks for u == NaN or u == inf
        v_if(sfpi::as<sfpi::vInt>(u) >= sfpi::as<sfpi::vInt>(infinity)) { r = u; }
        v_endif;
    }
    v_endif;

    return r;
}

// Fast bf16 log1p for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 506.1 cycles/tile vs 1074.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// SFPU state programmed by the init: programmable constants LREG11 = -1.0f (its default value), LREG12-14 =
// C0, C1, P0 (via SFPCONFIG); scratch LREG5-7 = -4.0f, -0.25f, 4/3 (plain SFPLOADI, assumed to survive between
// faces/tiles -- any other SFPU op between log1p tiles needs a log1p re-init); SFPLOADMACRO InstructionTemplate
// [0..3] (T0 round, T1 setman, T2 w-MAD, T3 ma-IADD), LoadMacroConfig.Sequence[0..3] (macros 0..3) + Misc;
// ADDR_MOD_6 = dest incr 2 (advances the RWC on every pipelined store). The per-face body records replay slots
// 0..14 (15 instructions, TTI_REPLAY execute-while-loading) and replays them. The software pipeline runs across
// the 4 faces of a tile and keeps a static face counter, so it requires VectorMode::RC (4 faces per call) -- the
// only mode log1p_tile uses. bf16 DEST only -- gated on !is_fp32_dest_acc_en below; the sfpi path stays as the
// fallback for every other configuration and for DISABLE_SFPLOADMACRO builds.
//
// Algorithm (all fp32 in LREGs; DST holds bf16):
//   w  = a*(4/3') + 4/3'          w in [1,2)*2^k  <=>  1+a in ~[0.7485,1.497)*2^k
//   e' = setman(w, 0)             biased bucket (127+k)<<23
//   n  = as_int(1.0f) - e'        = -(k<<23)   (bias cancels)
//   nf = float(n)                 via 2s-comp->sign-mag cast, then sign-mag->fp32 cast
//   ma = as_int(a) + n            = 2^-k * a   (exact scaling; denormal bits preserved)
//   s4 = as_int(-4.0f) + n        = -4 * 2^-k  (-4 base keeps k=128 in normal range)
//   t  = -0.25*s4 - 1             = 2^-k - 1   (exact / correctly rounded)
//   m  = ma + t                   = 2^-k*(1+a) - 1 in ~[-0.2515, 0.4971), exact reduction
//   res = (C1*m + C0)*m^2 + ((-nf)*P0 + m)     P0 = ln2 * 2^-23 (n is in exponent-bit units)
//   res = a*0 + res               inf fix: finite a -> +0 (exact no-op);
//                                 a=+inf -> NaN, and SFPSTOCHRND maps +NaN -> +inf
//   out = stochrnd_rne_bf16(res)  round-to-nearest (ties away) fp32 -> bf16
//
// x < -1 and NaN inputs are don't-care (golden NaN). x = -1 falls out as -inf via the
// same bit-level path (m -> -huge, m*m -> +inf, poly -> -inf).
//
// Implementation: software-pipelined at II = 15 issues/vector using SFPLOADMACRO.
// Three loadmacros per vector carry 5 extra scheduled ops (w-MAD, setman, ma-IADD,
// round, store), so the 20 SFPU ops per vector need only 15 issue slots, with all
// MAD latencies hidden by the pipeline (no stalls by construction).
//
// Steady-state frame (issue slots; scheduled fires in [brackets]):
//   0: A(v)=loadmacro0 L0,+2   [w @1, setman @3]
//   1: B(v)=loadmacro1 L1,+2   [ma @5]
//   2: res(v-1)                (p1*s + kt)
//   3: C(v-1)=loadmacro2 L2,+0 [rnd @7 -> L16, store @8, dest += 2]
//   4: n(v)
//   5: fix(v-1)                (a*0 + res)
//   6: cast1(v)  7: s4(v)  8: t(v)  9: cast2(v)  10: m(v)
//   11: nop  12: p1(v)  13: kt(v)  14: s(v)
//
// Verified bit-exact against a software model of the full bf16 sweep (max 1 ULP vs
// float64 golden; gate is 2 ULP).
#ifndef DISABLE_SFPLOADMACRO
inline void _init_log1p_bf16_fast_() {
    // Common SFPU init (config reg + ADDR_MOD_7 + counter reset) inlined so this init is self-contained, plus the
    // dest-increment-2 ADDR_MOD_6 the pipelined stores advance the RWC with (the generic unary init does not
    // program ADDR_MOD_6 for SfpuType::log1p).
    _init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Programmable constants (LREG11-14).
    _sfpu_load_config32_(11, 0xBF80, 0x0000);  // -1.0f
    _sfpu_load_config32_(12, 0xBF02, 0x5990);  // C0 = -0.5091791f   (poly: log1p(m) ~ m + m^2*(C0 + C1*m))
    _sfpu_load_config32_(13, 0x3E8F, 0x07FB);  // C1 =  0.27935776f
    _sfpu_load_config32_(14, 0x33B1, 0x7218);  // P0 = ln2 * 2^-23

    // Scratch-bank constants (persist across calls; nothing else touches LREGs).
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0xC080);  // -4.0f (bits used as int base)
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, 0xBE80);  // -0.25f
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x3FAB);  // ~4/3 (1.3359375)

    // --- SFPLOADMACRO templates (backdoor write: executing with VD=12+i) ----
    TTI_SFP_STOCH_RND(0, 0, 2, 2, 12, 1);                // T0: rnd fp32->bf16 RNE
    TTI_SFPSETMAN(0, 2, 13, 1);                          // T1: setman(x, 0)
    TTI_SFPMAD(p_sfpu::LREG7, 2, p_sfpu::LREG7, 14, 0);  // T2: w = (4/3)*x + 4/3
    TTI_SFPIADD(0, p_sfpu::LREG0, 15, 4);                // T3: ma = n(L0) + x

    // Macro 0 (A): mad = T2 delay 0 VB<-VD (0x80|0x06=0x86);
    //              simple = T1 delay 2 VC<-VD (0x00|0x10|0x05=0x15)
    TTI_SFPCONFIG(0x8615, 4 + 0, 1);
    // Macro 1 (B): simple = T3 delay 3 VB<-VD (0x80|0x18|0x07=0x9F)
    TTI_SFPCONFIG(0x009F, 4 + 1, 1);
    // Macro 2 (C): round = T0 delay 3 VC<-VD dst L16 (0x40|0x18|0x04=0x5C);
    //              store = builtin delay 4 from L16 (0x40|0x20|0x03=0x63)
    _sfpu_load_config32_(4 + 2, 0x635C, 0x0000);
    // Macro 3 (C-epilogue): same as macro 2 with tighter delays (round delay 2,
    // store delay 3) for the shortened per-face epilogue.
    // round = 0x40|(2<<3)|4 = 0x54 ; store = 0x40|(3<<3)|3 = 0x5B
    _sfpu_load_config32_(4 + 3, 0x5B54, 0x0000);
    // Misc: UnitDelayKind = 0xF (count issued SFPU instructions),
    //       UsesLoadMod0ForStore = macros 2+3, StoreMod0 = 0.
    TTI_SFPCONFIG(0xFC0, 8, 1);
    TTI_SFPNOP;
}

// One steady-state frame: head of vector v interleaved with tail of vector v-1.
// Invariant: at slot 0 the dest RWC points at vector v; C(v-1) accesses RWC-2
// (imm10 = 0x3FE wraps mod 1024) and advances the RWC by 2 for the next frame.
#define LOG1P_FAST_FRAME()                                                                 \
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_7, 0); /* A(v): a->L0 @+0; w, setman */     \
    TTI_SFPLOADMACRO((1 << 2) | 1, 0, ADDR_MOD_7, 0); /* B(v): a->L1 @+0; ma */            \
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG4, 0); /* res */   \
    TTI_SFPLOADMACRO((2 << 2) | 2, 0, ADDR_MOD_6, 0x3FE); /* C(v-1): a @-2; rnd, store */  \
    TTI_SFPIADD(0, p_sfpu::LCONST_1, p_sfpu::LREG0, 6);                    /* n */         \
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG4, p_sfpu::LREG2, 0); /*fix*/  \
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG3, 3);                          /* cast1 */     \
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, 4);                       /* s4 */        \
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG0, p_sfpu::LREG11, p_sfpu::LREG0, 0); /* t */    \
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);                          /* cast2 */     \
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, 0); /* m */  \
    TTI_SFPNOP;                                                            /* spare */     \
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG12, p_sfpu::LREG4, 0); /* p1 */  \
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG1, p_sfpu::LREG3, 1);  /* kt */  \
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0) /* s */

// One face (8 dst vectors). The software pipeline runs continuously across the 4 faces of a tile: face 0 fills
// it (prologue), faces 1-3 run steady frames whose first frame carries the previous face's last-vector tail,
// and face 3 drains it. The static face counter therefore assumes every call processes exactly 4 faces in
// order (VectorMode::RC).
inline void _calculate_log1p_bf16_fast_() {
    static int log1p_fast_face = 0;

    if (log1p_fast_face == 0) {
        // Prologue frame (v0): no v-1 tail. Slot 3 hosts a dummy load whose only
        // job is the ADDR_MOD_6 advance a C-macro would have done (keeps the
        // "RWC points at v at slot 0" invariant for the following frames).
        TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_7, 0);  // A(v0)
        TTI_SFPLOADMACRO((1 << 2) | 1, 0, ADDR_MOD_7, 0);  // B(v0)
        TTI_SFPNOP;                                        // (res slot)
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_6, 0);      // (C slot: dummy, advance +2)
        TTI_SFPIADD(0, p_sfpu::LCONST_1, p_sfpu::LREG0, 6);
        TTI_SFPNOP;  // (fix slot)
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG3, 3);
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, 4);
        TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG0, p_sfpu::LREG11, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG12, p_sfpu::LREG4, 0);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG1, p_sfpu::LREG3, 1);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);

        // 7 steady frames (v1..v7): record once, replay 6x.
        TTI_REPLAY(0, 15, 1, 1);
        LOG1P_FAST_FRAME();
#pragma GCC unroll 8
        for (int d = 2; d < 8; d++) {
            TTI_REPLAY(0, 15, 0, 0);
        }
    } else {
        // 8 steady frames; the first one carries the previous face's v7 tail.
        TTI_REPLAY(0, 15, 1, 1);
        LOG1P_FAST_FRAME();
#pragma GCC unroll 8
        for (int d = 1; d < 8; d++) {
            TTI_REPLAY(0, 15, 0, 0);
        }
        if (log1p_fast_face == 3) {
            // Drain: tail of the tile's last vector, macro 3 (rnd delay 2, store 3).
            TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG4, 0);
            TTI_SFPLOADMACRO((3 << 2) | 2, 0, ADDR_MOD_6, 0x3FE);  // C(v7): a @-2
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG4, p_sfpu::LREG2, 0);
            TTI_SFPNOP;  // delay tick
            TTI_SFPNOP;  // rnd fires here
            TTI_SFPNOP;  // store fires here
        }
    }
    log1p_fast_face = (log1p_fast_face + 1) & 3;
}

#undef LOG1P_FAST_FRAME
#endif  // DISABLE_SFPLOADMACRO

/**
 * @tparam APPROXIMATION_MODE Ignored
 * @tparam FAST_APPROX Ignored
 * @tparam is_fp32_dest_acc_en If true, DEST registers are fp32, and output does not need to be rounded to bfloat16
 * @tparam ITERATIONS Number of iterations for given face
 */
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_log1p() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !FAST_APPROX && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_log1p_bf16_fast_();
        return;
    }
    if constexpr (
        !APPROXIMATION_MODE && !FAST_APPROX && !is_fp32_dest_acc_en &&
        !(!APPROXIMATION_MODE && !FAST_APPROX && !is_fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8 (tt-llk tests): log1p_init cannot see ITERATIONS and has programmed the fast kernel's
        // C0 / C1 / P0 into LREG12..14 (vConstFloatPrgm0..2) on top of the constants calculate_log1p_fp32 reads;
        // re-seed them before falling back. The rest of the fast state is inert here: LREG11 was written with its
        // default -1.0f, LREG5..7 are sfpi-allocated scratch, the SFPLOADMACRO templates/sequences are unused by
        // the sfpi body, and ADDR_MOD_6 (dest incr 2) is never referenced by sfpi dst_reg code (ADDR_MOD_7).
        _init_log1p_body_constants_<is_fp32_dest_acc_en>();
    }
#endif
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat result = calculate_log1p_fp32<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Production constants read by calculate_log1p_fp32 (log1p and body re-users: asinh/acosh/atanh). Body re-users
// call this directly: log1p_init below programs the fast bf16 kernel's state instead on its gate.
template <bool is_fp32_dest_acc_en>
inline void _init_log1p_body_constants_() {
    const float LOG_TWO = 0.693147182f;       // 0x1.62e430p-1
    const float TWO_TO_M23 = 1.19209290e-7f;  // 0x1.0p-23
    // e represents k << 23 rather than k, so pre-fold the 2^(-23) factor into
    // the constant used for the final exponent contribution.
    sfpi::vConstFloatPrgm0 = LOG_TWO * TWO_TO_M23;

    if constexpr (is_fp32_dest_acc_en) {
        // Stored separately because the tuned fp32 m^3 and m^4 coefficients are
        // no longer the shared exact 1/3 and -1/4 values used in the bf16 path.
        sfpi::vConstFloatPrgm1 = -0x1.00001ap-2f;
        sfpi::vConstFloatPrgm2 = 0x1.555572p-2f;
    } else {
        // Horner coefficients used by bf16 polynomial
        sfpi::vConstFloatPrgm1 = 0x1.744p-2f;
        sfpi::vConstFloatPrgm2 = -0x1.008p-1f;
    }
}

/**
 * @tparam APPROXIMATION_MODE Ignored
 * @tparam FAST_APPROX Ignored
 * @tparam is_fp32_dest_acc_en If true, DEST registers are fp32, and output does not need to be rounded to bfloat16
 */
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log1p_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !FAST_APPROX && !is_fp32_dest_acc_en) {
        _init_log1p_bf16_fast_();
        return;
    }
#endif
    _init_log1p_body_constants_<is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
