// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat _sfpu_exp2_fp32_accurate_(sfpi::vFloat x) {
    sfpi::vFloat f, j, r, y, abs_x;
    sfpi::vInt i;
    sfpi::vSMag16 sm;

    // Convert x to sign-magnitude 16-bit integer (round to nearest with ties
    // away from zero), and convert back to floating point.
    sm = sfpi::convert<sfpi::vSMag16>(x, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);

    // Range reduced value in [-0.5, 0.5].
    f = x - j;

    // Minimax polynomial approximation for exp2(f), f in [-0.5, 0.5].
    // The first three coefficients are rounded to fp16 (one sfploadi per coefficient).
    // The subsequent three coefficients are fp32 values stored in constant registers.
    // Interleaved with conversion of sign-magnitude integer in [-32767, 32767] to two's complement.
    // Interleaved with calculation of the overflow case y = y * inf, giving inf or NaN.
    // Interleaved with calculation of abs_x = abs(x), which is >= 0.0 unless x is -NaN.
    r = 0x1.41cp-13f;
    r = r * f + 0x1.5f4p-10f;
    r = r * f + 0x1.3b4p-7f;
    i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
    y = r * f + sfpi::vConstFloatPrgm2;
    i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), j));
    r = y * f + sfpi::vConstFloatPrgm1;
    y *= std::numeric_limits<float>::infinity();
    r = r * f + sfpi::vConstFloatPrgm0;
    abs_x = sfpi::abs(x);
    r = r * f + 1.0f;

    // Exclude -NaN: abs(-NaN) remains negative.
    v_if(abs_x >= 0.0f) {
        sfpi::vInt e = sfpi::exexp(r, sfpi::ExponentMode::Biased);
        e += i;
        // e < 255
        v_block {
            sfpi::vInt e_lt_255 = __builtin_rvtt_sfpiadd_i(e.get(), -255, sfpi::SFPIADD_MOD1_CC_LT0);
            y = sfpi::setexp(r, e);
            // e < 1
            v_if(e_lt_255 < -254) {
                // Underflow, including subnormals.
                y = 0.0f;
            }
            v_endif;
        }
        v_endblock;
    }
    v_endif;

    return y;
}

// BF16 path: branch-free saturation using the mantissa-as-fractional-part trick
// from the production `_sfpu_exp_21f_bf16_` kernel — but specialised for exp2 by
// skipping the `* (1/ln2)` multiply (we are already in base-2). vec_min_max clamps
// xlog2 to [0, 255], so the natural saturation of setexp + the final bf16 round
// give the correct overflow → +inf and underflow → 0 boundary encodings for free.
//
// NaN: ttnn.bfloat16 host-side packing already collapses NaN → +inf before the
// tensor ever reaches the SFPU (see the "NaN is packed as inf for ttnn.bfloat16"
// xfails on fmod / remainder / where / rdiv), so a device-side NaN guard here
// would be dead code.
sfpi_inline sfpi::vFloat _sfpu_exp2_bf16_(sfpi::vFloat x) {
    // Map x → xlog2 such that 2^x has biased exponent floor(xlog2) and the
    // fractional mantissa supplies the (xlog2 - floor) refinement.
    sfpi::vFloat xlog2 = x + 127.f;

    // Clamp to [0, 255]. Boundary inputs land on the +inf / +0 encodings after
    // setexp + bf16 round.
    xlog2 = sfpi::clamp(xlog2, 0.0f, 255.0f);

    // Decompose xlog2 in [0, 255] into:
    //   exponential_part = floor(xlog2)             (integer in [0, 255])
    //   fractional_part  = (xlog2 - floor) * 2^23   (integer in [0, 2^23))
    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);

    sfpi::vInt exponential_part = sfpi::exexp(sfpi::as<sfpi::vFloat>(z), sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part = sfpi::exman(sfpi::as<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);

    // Refine 2^x_f on x_f to [0, 2^23). Same minimax coefficients as the
    // production exp_21f kernel (≤ 3 fp32 ULP, well under 1 bf16 ULP).
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: 2^x = (1.frac_mantissa) * 2^(exponential_part - 127).
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    // SFPSTORE truncates fp32→bf16; round explicitly so the bf16 result matches
    // a faithful nearest-even rounding of the fp32 mathematical value, and so
    // that the saturation tricks above (overflow → +inf, underflow → 0) land
    // on the correct bf16 encoding.
    return sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
}

// Fast bf16 exp2 for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-opus-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 430.12 cycles/tile on p150b (tt-metal v0.76.0 baseline; the bench manifest records no production
// reference cycle count for exp2).
//
// Selected by exp2_init / calculate_exp2 for bf16 dest and ITERATIONS = 8. Not gated on APPROXIMATION_MODE: the
// compute API hard-codes APPROXIMATE = true for exp2 and the production bf16 kernel ignores the parameter.
// SFPU state programmed by _init_exp2_bf16_fast_(): programmable constants vConstFloatPrgm0..2 (LREG12..14) only.
// Pure sfpi: no SFPLOADMACRO, no replay slots, no ADDR_MOD_6 (dst_reg[0..7] relative to the face base, i.e. the
// common init prologue's ADDR_MOD_7 with dest increment 0).
//
//   t = x + 127, floored at 0        (biased-exponent domain)
//   n = (uint8)trunc(t)              saturates at 255 -> gives the +inf case free
//   f = t - n                        in [0, 1)
//   2^x = P(f) * 2^(n-127)
//
// P is a minimax quadratic for 2^f on [0,1], nudged up by ~3/4 of a bf16 ulp so
// that the truncating SFPSTORE behaves like round-to-nearest (no extra
// SFPSTOCHRND needed).  2^(n-127) is a single SFPSETEXP on the constant 1.0:
// n = 255 -> +inf, n = 0 -> +0, so both saturations come out of the same
// instruction.
//
// 11 SFPU instructions per vector (incl. load/store).  The loop body handles
// two vectors at a time: the chain is fully serial, so interleaving two
// independent chains hides the per-instruction latency (2 vectors is the most
// that fits without spilling LREGs).

// Minimax (relative error) quadratic for 2^f on f in [0, 1], scaled by 1.0011.
#define EXP2_FAST_C0 1.0028266704579617f
#define EXP2_FAST_C1 0.6583596756195741f
#define EXP2_FAST_C2 0.3375603430976409f

inline void _init_exp2_bf16_fast_() {
    sfpi::vConstFloatPrgm0 = EXP2_FAST_C0;
    sfpi::vConstFloatPrgm1 = EXP2_FAST_C1;
    sfpi::vConstFloatPrgm2 = EXP2_FAST_C2;
}

#undef EXP2_FAST_C0
#undef EXP2_FAST_C1
#undef EXP2_FAST_C2

inline void _calculate_exp2_bf16_fast_() {
    constexpr size_t vectors_per_face = 8;
#pragma GCC unroll 4
    for (size_t i = 0; i < vectors_per_face; i += 2) {
        sfpi::vFloat t0 = sfpi::dst_reg[i];
        sfpi::vFloat t1 = sfpi::dst_reg[i + 1];
        t0 = t0 + 127.0f;
        t1 = t1 + 127.0f;
        t0 = sfpi::max(t0, 0.0f);
        t1 = sfpi::max(t1, 0.0f);

        // sfpi >= 7.80: float_to_uint8 was removed; convert<vUInt8> emits the same SFPSTOCHRND uint8 conversion.
        sfpi::vMag n0 = sfpi::convert<sfpi::vUInt8>(t0, sfpi::RoundMode::Zero);
        sfpi::vMag n1 = sfpi::convert<sfpi::vUInt8>(t1, sfpi::RoundMode::Zero);
        sfpi::vFloat f0 = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(n0), sfpi::RoundMode::Nearest);
        sfpi::vFloat f1 = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(n1), sfpi::RoundMode::Nearest);
        f0 = t0 - f0;
        f1 = t1 - f1;

        sfpi::vFloat p0 = sfpi::vConstFloatPrgm2 * f0 + sfpi::vConstFloatPrgm1;
        sfpi::vFloat p1 = sfpi::vConstFloatPrgm2 * f1 + sfpi::vConstFloatPrgm1;
        p0 = p0 * f0 + sfpi::vConstFloatPrgm0;
        p1 = p1 * f1 + sfpi::vConstFloatPrgm0;
        sfpi::vFloat s0 = sfpi::setexp(sfpi::vFloat(1.0f), sfpi::as<sfpi::vInt>(n0));
        sfpi::vFloat s1 = sfpi::setexp(sfpi::vFloat(1.0f), sfpi::as<sfpi::vInt>(n1));

        sfpi::dst_reg[i] = p0 * s0;
        sfpi::dst_reg[i + 1] = p1 * s1;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_exp2() {
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_exp2_bf16_fast_();
        return;
    }
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        if constexpr (is_fp32_dest_acc_en) {
            sfpi::dst_reg[0] = _sfpu_exp2_fp32_accurate_(v);
        } else {
            sfpi::dst_reg[0] = _sfpu_exp2_bf16_(v);
        }

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void exp2_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (is_fp32_dest_acc_en) {
        // Coefficients for minimax polynomial.
        sfpi::vConstFloatPrgm0 = 0x1.62e42ep-1f;
        sfpi::vConstFloatPrgm1 = 0x1.ebfba0p-3f;
        sfpi::vConstFloatPrgm2 = 0x1.c6afd8p-5f;
    } else {
        // Fast bf16 kernel constants (see _calculate_exp2_bf16_fast_). The bf16 fallback for ITERATIONS != 8
        // (_sfpu_exp2_bf16_) uses immediates only, so it is unaffected by these.
        _init_exp2_bf16_fast_();
    }
}

}  // namespace ckernel::sfpu
