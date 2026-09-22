// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_polyval.h"

/*
 * The expm1(x) code is derived from code by Norbert Juffa.
 *
 * Copyright (c) 2015-2023 Norbert Juffa
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace ckernel::sfpu {

/*
 * i = rint(a / log(2)), f = a - i * log(2). Then
 * expm1(a) = 2**i * (expm1(f) + 1) - 1.
 *
 * Compute r = expm1(f). Then
 * expm1(a) = 2 * (0.5 * 2**i * r + 0.5 * 2**i - 0.5).
 *
 * With t = 0.5 * 2**i, expm1(a) = 2 * (r * t + t - 0.5).
 * For best accuracy, use expm1(a) = 2 * (r + 0.5) when i == 1,
 * and expm1(a) = r when i == 0.
 *
 * This approach avoids underflow for tiny values, and overflow for huge
 * values.
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_expm1_(sfpi::vFloat a) {
    sfpi::vFloat log2e = sfpi::vConstFloatPrgm0;
    sfpi::vFloat rounding_bias = 12582912.f;
    sfpi::vFloat j = __builtin_rvtt_sfpmad(log2e.get(), a.get(), rounding_bias.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    sfpi::vFloat r;

    if constexpr (!is_fp32_dest_acc_en) {
        sfpi::vFloat scale, bias;

        r = 8.361816406e-03f;
        sfpi::vInt i = sfpi::as<sfpi::vInt>(j);
        j = j - rounding_bias;

        sfpi::vFloat c2 = 4.177856445e-02f;

        sfpi::vFloat f = j * sfpi::vConstFloatPrgm1 + a;  // -ln(2)

        r = r * f + c2;

        sfpi::vFloat s = f * f;

        r = r * f + sfpi::vConstFloatPrgm2;

        sfpi::vFloat w = 0.5f;
        r = __builtin_rvtt_sfpmad(r.get(), f.get(), w.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        sfpi::vFloat infinity = std::numeric_limits<float>::infinity();

        r = r * s + f;

        // For j == 0.0, r is already expm1(a). Avoid half-scaled
        // reconstruction as subnormals flush to zero, so
        // (0.5 * r) * 2 can lose tiny normal results.
        v_if(j != 0.0f) {
            sfpi::vFloat jm2 = j + -2.0f;
            // Keep reconstruction half-scaled: scale is 0.5 * 2**i. Avoids
            // materialising 2**i directly near overflow boundary.
            scale = sfpi::as<sfpi::vFloat>((i << 23) + sfpi::as<sfpi::vInt>(w));

            sfpi::vFloat abs_jm2 = sfpi::abs(jm2);
            bias = scale - w;
            sfpi::vInt tail = sfpi::as<sfpi::vInt>(sfpi::convert<sfpi::vSMag8>(abs_jm2, sfpi::RoundMode::Nearest));
            r = scale * r + bias;

            v_if(tail >= 127) {
                // Positive side becomes +inf; NaNs should propagate through the multiply.
                r = jm2 * infinity;

                v_if(jm2 < 0.0f) { r = -0.5f; }
                v_endif;
            }
            v_endif;
            r *= 2.0f;
        }
        v_endif;
    } else {
        sfpi::vFloat s, t, u, x, y;

        r = 1.974105835e-04f;
        sfpi::vInt i = sfpi::as<sfpi::vInt>(j);
        j = j - rounding_bias;

        sfpi::vFloat c4 = 1.393107930e-3f;

        sfpi::vFloat f = j * sfpi::vConstFloatPrgm1 + a;
        f = j * -1.42860677e-6f + f;  // -ln(2)_lo

        s = f * f;

        r = r * f + c4;
        r = r * f + 8.333439939e-3f;
        r = r * f + 4.166680202e-2f;
        sfpi::vFloat w = 0.5f;
        r = r * f + sfpi::vConstFloatPrgm2;
        sfpi::vFloat c0 = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(w) + -1);

        u = f;
        sfpi::vFloat jm1 = j + -1.0f;
        r = r * f + c0;
        v_if(jm1 == 0.0f) { u += 0.5f; }
        v_endif;
        r = r * s + u;

        v_if(j != 0.0f) {
            v_if(jm1 != 0.0f) {
                t = sfpi::as<sfpi::vFloat>((i << 23) + sfpi::as<sfpi::vInt>(w));
                y = t - w;
                sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
                x = t - y;  // double-float canonicalization of difference
                sfpi::vFloat jm2 = jm1 + -1.0f;
                x = x - w;
                // abs(-NaN) = -NaN, otherwise the result will be positive.
                sfpi::vFloat abs_jm2 = sfpi::abs(jm2);
                r = r * t + x;
                // This will be -127 in the case of -NaN, otherwise 0 <= clamped <= 127.
                sfpi::vInt clamped =
                    sfpi::as<sfpi::vInt>(sfpi::convert<sfpi::vSMag8>(abs_jm2, sfpi::RoundMode::Nearest));
                r += y;
                // Handle special cases a * log2(e) <= -125 and a * log2(e) >= 129.
                v_if(clamped >= 127) {
                    // Positive case; multiply by j-2 to propagate NaN
                    r = jm2 * infinity;
                    v_if(jm2 < 0.0f) {
                        // Negative case; result will be -1 (note: -NaN was excluded earlier).
                        r = -0.5f;
                    }
                    v_endif;
                }
                v_endif;
            }
            v_endif;
            r *= 2.0f;
        }
        v_endif;
    }

    return r;
}

// Constants for the production expm1 kernel (_sfpu_expm1_): vConstFloatPrgm0..2 (LREG12..14). Programmed by
// expm1_init when the fast bf16 kernel below is not selected, and re-armed by calculate_expm1 when it must fall
// back to _sfpu_expm1_ with the fast kernel's constants installed (ITERATIONS != 8).
template <bool is_fp32_dest_acc_en>
inline void _init_expm1_prgm_constants_() {
    sfpi::vConstFloatPrgm0 = 1.442695f;  // log2(e) == 1 / ln(2)
    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = -0.693145752f;    // -ln(2)_hi
        sfpi::vConstFloatPrgm2 = 1.666667163e-1f;  // c1
    } else {
        sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // -ln(2)
        sfpi::vConstFloatPrgm2 = 1.666259766e-01f;      // c1
    }
}

// Fast bf16 expm1 for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 487.13 cycles/tile vs 1231.13 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
//
// Selected by expm1_init / calculate_expm1 for APPROXIMATION_MODE = false, bf16 dest, ITERATIONS = 8
// (SFPLOADMACRO builds only).
// SFPU state programmed by _init_expm1_bf16_fast_(): programmable constants LREG11 (SFPCONFIG dest 11) and
// vConstFloatPrgm0..2 (LREG12..14), programmable macro instruction 0 (mux 4, SFPCONFIG dest 0), macro sequence
// register 1 (SFPCONFIG dest 5) and LoadMacroConfig.Misc. No replay slots, no ADDR_MOD_6: dst is addressed with
// explicit offsets 0,2,..,14 via ADDR_MOD_7 (dest increment 0, as programmed by the common init prologue).
// LREG5 / LREG6 (poly coefficients c1 / c2) are plain LREGs and are (re)loaded at the top of every
// _calculate_expm1_bf16_fast_() call: the bench loaded them once in init, but plain LREGs do not survive other
// SFPU ops issued between expm1_tile_init and expm1_tile in fused production kernels.
//
// Algorithm (exhaustively validated: all 65536 bf16 inputs, max 1 ULP vs
// float64 golden, on silicon):
//
//   x   = max(x, -88)          // SFPLOADMACRO: LD + SWAP vs creg14; keeps the
//                              // reduced argument f finite for deep negatives
//                              // (expm1 = -1 there); -inf handled here too.
//   y   = x*L + 127            // L = 1.4405, deliberately < log2(e): keeps
//                              // x = 88.5 in the k=127 (finite) bin while
//                              // x = 89 rounds to k=128 -> +inf (golden
//                              // overflows there as well).
//   k8  = round_sat_u8(y)      // SFPSTOCHRND FP32->UINT8 saturates to
//                              // [0, 255], so huge x -> k8 = 255 -> 2^k = inf
//                              // and clamped negatives -> k8 >= 0. Replaces
//                              // the magic-bias trick AND the upper clamp.
//   2^k = as_float(k8 << 23)   // k8 = k + 127 is exactly the exponent field;
//                              // k8 = 0 gives +0 (deep negative lanes).
//   kf  = cast(k8) - 127
//   f   = kf*(-ln2) + x        // exact passthrough f = x when k = 0, so no
//                              // catastrophic cancellation near zero.
//   p   = ((c2 f + c1) f + 1) f            // expm1(f), weighted-minimax fit
//                                          // on [-0.50, 0.50] (rel err ~2e-4)
//   res = 2^k * p + (2^k - 1)  // 2^k - 1 kept separate: k = 0 lanes return p
//                              // exactly. k8 = 255 lanes have p > 0 so the
//                              // MAD gives +inf (0*inf = inf on this SFPU,
//                              // and k8 = 0 lanes have finite p thanks to
//                              // the -88 clamp).
//   store truncates fp32 -> bf16 (round-to-zero); combined error still <= 1
//   ULP of the RNE golden everywhere (verified exhaustively).
//
// Implementation: hand-scheduled TTI stream, software-pipelined two deep:
// each body computes element A, stores element A-1's result (parked in L2),
// and macro-loads + clamps element B = A+1 in a single SFPLOADMACRO issue.
// 13 issue slots per element, nearly stall-free. The SFPSWAP rides the
// LOADMACRO's SIMPLE-unit slot; empirically, the instruction issued
// immediately after a LOADMACRO must not be a MAD-unit op (the macro's empty
// MAD slot swallows it), so the store occupies that slot.
//
// Persistent state (programmed once in _init_expm1_bf16_fast_; SFPU constant and
// macro registers survive across faces and tiles):
//   creg11 = 127.0   creg12 = L = 1.4405   creg13 = -ln2   creg14 = -88.0
// Per-call state (loaded at the top of _calculate_expm1_bf16_fast_):
//   L5 = c1 = 0.50372314   L6 = c2 = 0.16637857
// Scratch per element:
//   L0 = clamped x     L1 = two_k          L2 = prev res -> q/h/p -> res
//   L3 = kf -> f       L4 = k8 -> next y   L7 = tm1 = 2^k - 1
inline void _init_expm1_bf16_fast_() {
    sfpi::vConstFloatPrgm0 = 1.4405f;               // creg12 = L
    sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // creg13 = -ln2
    sfpi::vConstFloatPrgm2 = -88.0f;                // creg14 = clamp bound

    // creg11 = 127.0f (y addend / kf subtrahend)
    TTI_SFPLOADI(p_sfpu::LREG0, 0, 0x42FE);
    TTI_SFPCONFIG(0, 11, 0);
    TTI_SFPNOP;

    // Programmable macro instruction 0 (mux4) = SFPSWAP(0, 0, 14, VEC_MIN_MAX)
    // = 0x920000E1: src_c = 0 is substituted with the loadmacro register, so
    // the larger of (loaded x, creg14 = -88) lands in the load register; the
    // min-side writeback to creg14 is dropped.
    TTI_SFPLOADI(p_sfpu::LREG0, 0xA, 0x00E1);
    TTI_SFPLOADI(p_sfpu::LREG0, 0x8, 0x9200);
    TTI_SFPCONFIG(0, 0, 0);
    TTI_SFPNOP;

    // Macro sequence register 1 (config dest 5): SIMPLE = SWAP(mux4) @ delay 0;
    // MAD/ROUND/STORE slots unused.
    TTI_SFPLOADI(p_sfpu::LREG0, 0xA, 0x0004);
    TTI_SFPLOADI(p_sfpu::LREG0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 5, 0);

    // Reset LoadMacroConfig misc; UnitDelayKind = 0xF guards pipeline races.
    TTI_SFPCONFIG(0xF00, 0x8, 0x1);
    TTI_SFPNOP;
}

// clang-format off
// Steady-state body: computes element A (clamped x in L0, k8 in L4), stores
// element A-1's result (in L2) to POFF, macro-loads + clamps element B from
// NOFF into L0, and produces B's y/k8 into L4. 13 issue slots.
#define EXPM1_FAST_BODY(POFF, OFF, NOFF)                                                    \
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG3, 0);            /* kfloat        */      \
    TTI_SFPSHFT(23, p_sfpu::LREG4, p_sfpu::LREG1, 7);        /* two_k         */      \
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG11, p_sfpu::LREG3, p_sfpu::LREG3, 1);    \
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG7, 1);  \
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG0, p_sfpu::LREG3, 0);       \
    TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, NOFF);                /* B.x -> L0     */      \
    TTI_SFPSTORE(p_sfpu::LREG2, 2, ADDR_MOD_7, POFF);        /* prev res      */      \
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG2, 0);        \
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG11, p_sfpu::LREG4, 0);      \
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);     \
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG4, p_sfpu::LREG4, 2);   /* B.k8 */         \
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);     \
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG2, 0); /* res */

// Last element of the face: no next-element prep; store its result inline so
// no macro or store is pending when the wrapper advances the dest counter.
#define EXPM1_FAST_BODY_LAST(POFF, OFF)                                                     \
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG3, 0);                                     \
    TTI_SFPSHFT(23, p_sfpu::LREG4, p_sfpu::LREG1, 7);                                 \
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG11, p_sfpu::LREG3, p_sfpu::LREG3, 1);    \
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG7, 1);  \
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG0, p_sfpu::LREG3, 0);       \
    TTI_SFPSTORE(p_sfpu::LREG2, 2, ADDR_MOD_7, POFF);                                 \
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG2, 0);        \
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);     \
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);     \
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG1, 0);        \
    TTI_SFPSTORE(p_sfpu::LREG1, 2, ADDR_MOD_7, OFF);
// clang-format on

inline void _calculate_expm1_bf16_fast_() {
    // Poly coefficients: L5 = c1 = 0x3F00F400, L6 = c2 = 0x3E2A5F25 (plain LREGs, hence per call; see above).
    TTI_SFPLOADI(p_sfpu::LREG5, 0xA, 0xF400);
    TTI_SFPLOADI(p_sfpu::LREG5, 0x8, 0x3F00);
    TTI_SFPLOADI(p_sfpu::LREG6, 0xA, 0x5F25);
    TTI_SFPLOADI(p_sfpu::LREG6, 0x8, 0x3E2A);

    // Prologue: macro-load + clamp element 0; produce its y and k8. The two
    // NOPs cover the macro SWAP's latency (macro writes are not interlocked
    // against issued instructions).
    TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, 0);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG11, p_sfpu::LREG4, 0);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG4, p_sfpu::LREG4, 2);
    // Body 0 stores stale L2 to offset 0 (element 0's input slot, already
    // consumed); body 1's POFF = 0 overwrites it with the real result.
    EXPM1_FAST_BODY(0, 0, 2);
    EXPM1_FAST_BODY(0, 2, 4);
    EXPM1_FAST_BODY(2, 4, 6);
    EXPM1_FAST_BODY(4, 6, 8);
    EXPM1_FAST_BODY(6, 8, 10);
    EXPM1_FAST_BODY(8, 10, 12);
    EXPM1_FAST_BODY(10, 12, 14);
    EXPM1_FAST_BODY_LAST(12, 14);
}

#undef EXPM1_FAST_BODY
#undef EXPM1_FAST_BODY_LAST

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_expm1() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        if constexpr (ITERATIONS == 8) {
            _calculate_expm1_bf16_fast_();
            return;
        }
        // expm1_init<false, false> is not templated on ITERATIONS and programmed the fast kernel's constants into
        // vConstFloatPrgm0..2: re-arm the _sfpu_expm1_ constants before falling back to it.
        _init_expm1_prgm_constants_<is_fp32_dest_acc_en>();
    }
#endif
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat y = _sfpu_expm1_<is_fp32_dest_acc_en>(x);
        if constexpr (!is_fp32_dest_acc_en) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void expm1_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        // Fast bf16 kernel (_calculate_expm1_bf16_fast_): LREG11..14 + macro instruction 0 + macro sequence 1.
        _init_expm1_bf16_fast_();
        return;
    }
#endif
    _init_expm1_prgm_constants_<is_fp32_dest_acc_en>();
}

}  // namespace ckernel::sfpu
