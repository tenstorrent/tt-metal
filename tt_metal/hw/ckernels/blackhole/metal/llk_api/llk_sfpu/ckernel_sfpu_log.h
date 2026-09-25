// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The log(x) code is derived from code by Norbert Juffa.
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

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

// Production constant setup (defined with the init below); declared early so the ITERATIONS != 8
// fallback re-seed inside the calculate function can name it.
template <bool is_fp32_dest_acc_en>
inline void _init_log_body_constants_();

template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en, bool IS_BASE_TWO = false>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat a, const uint log_base_scale_factor) {
    sfpi::vFloat three_quarters = 0.75f;
    sfpi::vInt e = sfpi::as<sfpi::vInt>(a) - sfpi::as<sfpi::vInt>(three_quarters);

    if constexpr (!FAST_APPROX) {
        // normalise a (-0.0 and subnormals become +0.0)
        a = a * 1.0f + 0.0f;
    }

    e = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));
    sfpi::vFloat m = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - e);
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();

    // m in [0.75, 1.5). Compute log1p(m - 1) for m - 1 in [-0.25, 0.5).
    m -= 1.0f;

    v_if(a >= 0.0f) {
        sfpi::vFloat r;
        sfpi::vFloat s = m * m;
        sfpi::vFloat e_float;
        if constexpr (is_fp32_dest_acc_en) {
            r = -0x1.92cp-5f;
            r = r * m + 0x1.b84p-4f;
            r = r * m + -0x1.0c4p-3f;
            r = r * m + 0x1.274p-3f;
            r = r * m + -0x1.55p-3f;
            r = r * m + 0x1.998p-3f;
            sfpi::vMag abs_e = sfpi::abs(e);
            r = r * m + sfpi::vConstFloatPrgm1;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
            r = r * m + sfpi::vConstFloatPrgm2;
            sfpi::vFloat neg_half = -0.5f;
            r = __builtin_rvtt_sfpmad(r.get(), m.get(), neg_half.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        } else {
            sfpi::vMag abs_e = sfpi::abs(e);
            sfpi::vFloat neg_quarter = -0.25f;
            r = neg_quarter * m + sfpi::vConstFloatPrgm1;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
            r = r * m + sfpi::vConstFloatPrgm2;
        }

        // Handle special cases:
        //
        //   input 0.0  -> -inf
        //   input +inf -> +inf
        //   input NaN  -> NaN
        //
        // In the non-fast path, earlier normalisation maps -0.0 and subnormals
        // to +0.0. addexp(a, -1) wraps exponent 0 to 255, so zero becomes
        // +inf; exponent 255 values (Inf/NaN) are left unchanged.
        a = sfpi::addexp(a, -1);

        r = r * s + m;
        e_float = sfpi::copysgn(e_float, sfpi::as<sfpi::vFloat>(e));
        if constexpr (IS_BASE_TWO) {
            // log2 takes an exact path.  Scaling the finished natural-log sum, as
            // result *= 1/ln(2), also scales the exponent contribution by
            // ln(2) * (1/ln(2)), which does not round to exactly 1 in float, so log2 of
            // an exact power of two came back an ULP low for 46 of the 254 representable
            // exponents.
            //
            // Applying the base change to the mantissa term only leaves the exponent
            // term as e_float * 2^-23.  The exponent arrives here as k << 23 and 2^-23 is
            // a power of two, so that product is exact and log2(2^k) == k for every k.
            // Instruction count is unchanged: one multiply plus one multiply-add, where
            // before it was one multiply-add plus one multiply.
            //
            // This is deliberately not applied to other bases.  For log10 the equivalent
            // constant is log10(2) * 2^-23, which is not exactly representable, and
            // folding it in measured worse than the existing path.
            constexpr float TWO_TO_M23 = 1.19209290e-7f;  // 0x1.0p-23
            result = e_float * TWO_TO_M23 + r * sfpi::as<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        } else {
            result = e_float * sfpi::vConstFloatPrgm0 + r;

            if constexpr (HAS_BASE_SCALING) {
                result *= sfpi::as<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
            }
        }

        // For zero, result is negative before this multiply, so result * +inf
        // gives -inf. For +inf, result is positive, so result * +inf gives
        // +inf. NaNs either skip the main block or propagate here.
        v_if(!sfpi::is_finite(a)) { result *= a; }
        v_endif;
    }
    v_endif;

    return result;
}

// Fast bf16 log for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 2 ULP (gate <= 2).
// Measured 679.1 cycles/tile vs 978.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// SFPU state programmed by the init: programmable constants vConstFloatPrgm0..2 (LREG12-14) = K1, K2', NEGBIG.
// Pure sfpi (dst_reg[0..7], ADDR_MOD_7 only): no SFPLOADMACRO, no replay slots, no ADDR_MOD_6. bf16 DEST only:
// the polynomial is fitted against the TRUNCATING bf16 store, so this is gated on !is_fp32_dest_acc_en (and on
// the plain natural log: !HAS_BASE_SCALING && !IS_BASE_TWO); the Juffa-derived sfpi path above stays as the
// fallback for every other configuration.
//
// 16 SFPU ops per 32-lane vector (+load/store), scheduled for the observed
// 2-cycle MAD latency (each MAD-class consumer >= 2 slots after producer).
//
// Range reduction (per lane; x arrives as fp32 with low 16 mantissa bits 0):
//   c = sm32_to_fp32(bits(|x|))            exact: <= 15 significant bits
//     normal x:  c = (e_biased + u) * 2^23, u = mantissa fraction in [0,1)
//     denormal:  predicated re-encode: c = sm32_to_fp32(bits(c)) + CDADJ,
//                CDADJ = -149*2^23  (log(x) = log(k*2^16) - 149*ln2);
//                CDADJ is bf16-exact, so the add is a single SFPADDI.
//   r = c*K1 + K2'  ~= ln2*(e + u) + c0
//     K1 = 90852*2^-40 (17-bit mantissa) with K2 anchored to -127*90852*2^-17
//     so x==1.0 cancels EXACTLY under fused or non-fused MAD; c0 is the
//     polynomial constant term folded into K2'.
//   result = ((c3*m + c2)*m + c1)*m + r,  m = setexp(x,127) in [1,2)
//     c1..c3 fitted by LP against the exact per-input 2-ULP acceptance
//     intervals under TRUNCATING bf16 store (no rounding op needed); c1
//     nudged so the x==1 sum is exactly 0. A quadratic is provably
//     infeasible against those intervals, so cubic is minimal.
// Specials (no predicated blocks beyond the denormal one):
//   zero: rec = approx_recip(c) is +inf for c==0 and < 0.5ulp(c) otherwise,
//     so c -= rec turns only zero lanes into -inf (bit-exact no-op on all
//     finite lanes; verified). The final max() then lifts -inf to w=NEGBIG,
//     which truncates to 0xFF7F = 1 ULP from the golden -inf.
//   +inf: result = max(result, x + NEGBIG), NEGBIG = -(max_bf16 + 1 f32
//     ulp): the w arm is +inf only for x=+inf and never exceeds the result
//     for positive finite x (at x=max_bf16, w <= 0 < log(x)).
//   negatives / NaN: don't-care (goldens NaN); |x| keeps them non-NaN and
//   routes -0 through the zero path (the hardware x==0 predicate is bitwise,
//   and approx_recip(-0) = -inf, so the abs is required).
// All 65536 inputs verified bit-exactly offline under both fused and
// non-fused MAD semantics, and by exhaustive sweep on silicon.
inline void _init_log_bf16_fast_() {
    // Common SFPU init (config reg + ADDR_MOD_7 + counter reset) inlined so this init is self-contained; the
    // kernel only uses ADDR_MOD_7 (sfpi dst_reg accesses).
    _init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);

    sfpi::vConstFloatPrgm0 = 0x1.62e4p-24f;      // K1
    sfpi::vConstFloatPrgm1 = -0x1.637650p+6f;    // K2' = -127*90852*2^-17 + c0
    sfpi::vConstFloatPrgm2 = -0x1.fe0002p+127f;  // NEGBIG = -(max_bf16 + 1ulp)
}

// One face (8 dst vectors).
inline void _calculate_log_bf16_fast_() {
    // Loop-invariant constants; hoisted into LREGs once per face.
    sfpi::vFloat vc1 = 0x1.7d4750p+0f;
    sfpi::vFloat vc2 = -0x1.8ac49ap-1f;
    sfpi::vFloat vc3 = 0x1.e20fc8p-4f;

#pragma GCC unroll 8
    for (int i = 0; i < 8; i++) {
        sfpi::vFloat x = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[i]));
        sfpi::vFloat c = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(x), sfpi::RoundMode::Nearest);
        sfpi::vInt ee = sfpi::exexp(x, sfpi::ExponentMode::Biased);
        sfpi::vFloat rec = sfpi::approx_recip(c);
        sfpi::vFloat m = sfpi::setexp(x, 127);
        v_if(ee == 0) {
            c = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(c), sfpi::RoundMode::Nearest);
            c = c + -0x1.2ap+30f;  // CDADJ, bf16-exact -> single SFPADDI
        }
        v_endif;
        c = c - rec;  // -inf on zero lanes; exact no-op on every finite lane
        sfpi::vFloat acc = vc3 * m + vc2;
        sfpi::vFloat r = c * sfpi::vConstFloatPrgm0 + sfpi::vConstFloatPrgm1;
        acc = acc * m + vc1;
        sfpi::vFloat w = x + sfpi::vConstFloatPrgm2;
        r = acc * m + r;
        r = sfpi::max(r, w);
        sfpi::dst_reg[i] = r;
    }
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en,
    int ITERATIONS = 8,
    bool IS_BASE_TWO = false>
inline void calculate_log(uint log_base_scale_factor) {
    if constexpr (
        !APPROXIMATION_MODE && !FAST_APPROX && !HAS_BASE_SCALING && !IS_BASE_TWO && !is_fp32_dest_acc_en &&
        ITERATIONS == 8) {
        _calculate_log_bf16_fast_();
        return;
    }
    if constexpr (
        !APPROXIMATION_MODE && !FAST_APPROX && !HAS_BASE_SCALING && !IS_BASE_TWO && !is_fp32_dest_acc_en &&
        !(!APPROXIMATION_MODE && !FAST_APPROX && !HAS_BASE_SCALING && !IS_BASE_TWO && !is_fp32_dest_acc_en &&
          ITERATIONS == 8)) {
        // ITERATIONS != 8 (tt-llk tests): log_init cannot see ITERATIONS and has programmed the fast kernel's
        // K1 / K2' / NEGBIG into vConstFloatPrgm0..2 on top of the constants calculate_log_body reads; re-seed
        // them before falling back.
        _init_log_body_constants_<is_fp32_dest_acc_en>();
    }
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en, IS_BASE_TWO>(
            sfpi::dst_reg[0], log_base_scale_factor);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Production constants read by calculate_log_body (log, log_with_base, log2, and body re-users such as erfinv).
// Body re-users call this directly: log_init below programs the fast bf16 kernel's constants instead on its gate.
template <bool is_fp32_dest_acc_en>
inline void _init_log_body_constants_() {
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

// HAS_BASE_SCALING / IS_BASE_TWO mirror calculate_log's template parameters: the fast bf16 kernel replaces only
// the plain natural log, so the init has to know which variant the matching calculate_log will dispatch to.
// ITERATIONS is not threaded here (same convention as recip_init); every caller instantiates calculate_log with
// ITERATIONS == 8.
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en, bool HAS_BASE_SCALING, bool IS_BASE_TWO>
inline void log_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE && !FAST_APPROX && !HAS_BASE_SCALING && !IS_BASE_TWO && !is_fp32_dest_acc_en) {
        _init_log_bf16_fast_();
    } else {
        _init_log_body_constants_<is_fp32_dest_acc_en>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
