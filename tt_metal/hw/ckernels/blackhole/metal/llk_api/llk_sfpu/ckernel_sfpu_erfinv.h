// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt_custom.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat x) {
    // Algorithm based on "A handy approximation for the error function and its inverse" by Sergei Winitzki (2008)
    // This approximation defines erfinv(x) as:
    // erfinv(x) = sqrt( - 2/(pi*a) - log(1 - x^2)/2 + sqrt( ( 2/(pi*a) + log(1 - x^2)) ^2 - 1/a log(1 - x^2)) )
    // Where a is a polynomial coefficient used in the approximation of the error function (and reused in inverse error
    // function)

    // Compute log(1 - x^2)
    sfpi::vFloat log_value = calculate_log_body<false, false, false>(1.0f - x * x, 0);

    // Paper sets a constant a = 0.147.
    // This constant is used to compute two constant expressions:
    constexpr float TwoPiA = -4.330746750799873f;   // -2 / (pi * a)
    constexpr float OneDivA = 6.802721088435375f;  // 1/a

    // tmp = -2 / (pi * a) - log(1 - x^2)/2
    sfpi::vFloat tmp = TwoPiA + -0.5f * log_value;

    // calculated_value = temp + sqrt( temp^2 - log_value / a)
    sfpi::vFloat calculated_value = tmp * tmp - log_value * OneDivA;
    sfpi::vFloat intermediate_result = sfpu_sqrt_custom<false>(calculated_value);
    calculated_value = tmp + intermediate_result;

    // result = sqrt(calculated_value)
    sfpi::vFloat result = sfpu_sqrt_custom<false>(calculated_value);

    return result;
}

// ======================================================================
// Fast bf16 erfinv for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 567.1 cycles/tile vs 2991.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
//
// erfinv(x) for bf16 as a single polynomial in w = -log2(1 - x^2).
//
// Because bf16 can't represent x closer to 1 than 0.99609375, w spans only
// [0, 7.003], so erfinv(x) = x * P(w) with one cubic P covers the whole
// domain -- no branch between central/tail regions and no sqrt.
//
//   un  = x*x - 1                  (single MAD; = -(1-x^2), exact to <= 2^-25)
//   eb  = biased_exponent(un)      (|un| in [0.0078, 1] -> eb in [120,127];
//                                   eb == 0 only for x = +-1)
//   n   = setexp(un, 127)          (= -m, m = mantissa in [1,2); setexp keeps sign,
//                                   the sign is absorbed by negating QG1)
//   v   = (QG2*n + QG1N)*n + ef    (= QG2 m^2 + QG1 m + float(eb); the quadratic
//                                   qg(m) ~ log2(m) + K has no constant term, so
//                                   float(eb) is the Horner tail addend directly)
//   y   = x * Pt(v)                (cubic Pt is P shifted: Pt(K + 127 - w) ~ P(w);
//                                   its coefficients need full f32 precision)
//
// The m == 1.0 anchor (reachable only for 1-x^2 in {0,1}) puts the tiny-x
// case (x^2 < 2^-25) at v = 128.6796 where Pt is constrained to ~0.8892
// instead of erfinv'(0) = 0.88623 -- deliberately high so erfinv(1.125*2^-126)
// clears the f32 denormal output flush (its golden rounds UP to the min
// normal bf16, which a flushed-denormal result could never reach).
//
// x = +-1 (un == +0, eb == 0): ef is forced to +2^127.9, so v ~ +3.4e38 and
// Horner evaluation of Pt (positive leading coeff) overflows to +inf,
// giving y = x * inf = +-inf.
//
// Pt is also pre-scaled by ~(1 + 2^-9): the final SFPSTORE truncates
// f32 -> bf16, and the pre-scale turns truncation into (approximately)
// round-to-nearest, replacing an explicit SFPSTOCHRND.
//
// Two dst vectors are processed per iteration with interleaved instruction
// streams: dependent SFPU ops have ~2-cycle latency, independent ~1 --
// interleaving makes the loop pure instruction-throughput-bound.
//
// Domain: |x| <= 1. The exhaustive validation treated |x| > 1 (mathematically undefined; the previous
// kernel produced NaN there) as don't-care, so the output for |x| > 1 is NOT preserved by construction.
//
// Pure sfpi: no SFPLOADMACRO, no replay slots. Init programs LREG12 (PT2), LREG13 (PT1), LREG14 (PT0);
// the kernel READS vConstNeg1 (LREG11) and requires it to hold its default -1.0. bf16 DEST only.
// Selected by calculate_erfinv / erfinv_init when !APPROXIMATION_MODE && !is_fp32_dest_acc_en.
// ======================================================================

constexpr float ERFINV_FAST_QG2 = -0.3447265625f;         // fp16-representable: 1-instr SFPLOADI
constexpr float ERFINV_FAST_QG1N = -2.0242984294891357f;  // -QG1: full f32, hoisted register
// Pt: cubic, fit with an equality constraint at v(tiny x) = 128.6796 for the
// denormal-boundary anchor, pre-scaled by 1.0019454956 for truncating store.
constexpr float ERFINV_FAST_PT3 = 0.0007785453926771879f;  // full f32, hoisted register
constexpr float ERFINV_FAST_PT2 = -0.29362931847572327f;   // vConstFloatPrgm0
constexpr float ERFINV_FAST_PT1 = 36.73822784423828f;      // vConstFloatPrgm1
constexpr float ERFINV_FAST_PT0 = -1523.4005126953125f;    // vConstFloatPrgm2

inline void _init_erfinv_bf16_fast_() {
    sfpi::vConstFloatPrgm0 = ERFINV_FAST_PT2;
    sfpi::vConstFloatPrgm1 = ERFINV_FAST_PT1;
    sfpi::vConstFloatPrgm2 = ERFINV_FAST_PT0;
}

// One face = 8 dst vectors.
inline void _calculate_erfinv_bf16_fast_() {
    sfpi::vFloat qg1n = ERFINV_FAST_QG1N;
    sfpi::vFloat pt3 = ERFINV_FAST_PT3;

#pragma GCC unroll 4
    for (size_t i = 0; i < 8; i += 2) {
        sfpi::vFloat xa = sfpi::dst_reg[i];
        sfpi::vFloat xb = sfpi::dst_reg[i + 1];
        // un = x^2 - 1 = -(1 - x^2): single MAD; exponent extraction is
        // sign-agnostic and setexp keeps the sign, so n = -m below and the
        // sign is absorbed by negating QG1 (v = (QG2*n - QG1)*n + ef).
        // (-1.0f literal: sfpi maps it onto the reserved LREG11 constant, which every SFPU init restores.)
        sfpi::vFloat ua = xa * xa - 1.0f;
        sfpi::vFloat ub = xb * xb - 1.0f;
        sfpi::vInt eba = sfpi::exexp(ua, sfpi::ExponentMode::Biased);
        sfpi::vInt ebb = sfpi::exexp(ub, sfpi::ExponentMode::Biased);
        sfpi::vFloat ma = sfpi::setexp(ua, 127);
        sfpi::vFloat mb = sfpi::setexp(ub, 127);
        sfpi::vFloat efa = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(eba), sfpi::RoundMode::Nearest);
        sfpi::vFloat efb = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(ebb), sfpi::RoundMode::Nearest);
        v_if(eba == 0) { efa = 3.3895314e38f; }
        v_endif;
        v_if(ebb == 0) { efb = 3.3895314e38f; }
        v_endif;
        sfpi::vFloat g2 = ERFINV_FAST_QG2;
        sfpi::vFloat qa = g2 * ma + qg1n;
        sfpi::vFloat qb = g2 * mb + qg1n;
        sfpi::vFloat va = qa * ma + efa;
        sfpi::vFloat vb = qb * mb + efb;
        sfpi::vFloat pa = pt3 * va + sfpi::vConstFloatPrgm0;
        sfpi::vFloat pb = pt3 * vb + sfpi::vConstFloatPrgm0;
        pa = pa * va + sfpi::vConstFloatPrgm1;
        pb = pb * vb + sfpi::vConstFloatPrgm1;
        pa = pa * va + sfpi::vConstFloatPrgm2;
        pb = pb * vb + sfpi::vConstFloatPrgm2;
        xa = sfpi::dst_reg[i];
        xb = sfpi::dst_reg[i + 1];
        sfpi::vFloat ya = xa * pa;
        sfpi::vFloat yb = xb * pb;
        sfpi::dst_reg[i] = ya;
        sfpi::dst_reg[i + 1] = yb;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_erfinv() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        _calculate_erfinv_bf16_fast_();
        return;
    }
    constexpr int ITERATIONS = 8;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = calculate_erfinv_body<false>(in);
        in = sfpi::dst_reg[0];  // reload due to register pressure
        sfpi::dst_reg[0] = sfpi::copysgn(result, in);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void erfinv_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // Production log constants for calculate_log_body (log_init would select the bf16 fast-log constants here).
    _init_log_body_constants_<false>();
    // Fast bf16 path: programs LREG12-14 after log_init so its constants win (see _init_erfinv_bf16_fast_).
    // The common prologue (SFPU config reg + ADDR_MOD_7) is run by the llk_math_eltwise_unary_sfpu_init
    // callback overload before this function is called.
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        _init_erfinv_bf16_fast_();
    }
}

}  // namespace sfpu
}  // namespace ckernel
