// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel::sfpu {

// ======================================================================
// Shared machinery for the i1 asymptotic path, kept apart so that i0 can reuse
// it once it gains one (ckernel_sfpu_i0.h is a single polynomial today).
//
// Past |x| > 10 the asymptotic has the shape
//   i_n(|x|) ≈ exp(|x|) / sqrt(|x|) · P(1/|x|)
// where P is a degree-5 minimax fit specific to the order n. An i0 caller would
// differ only in the coefficient set and in skipping i1's final sign fix-up
// (i0 is even, i1 is odd).
//
// exp(|x|) leaves FP32 at 88.72284 but i0/i1 do not until ≈91.90 — the
// asymptotic value carries a 1/sqrt(2·pi·|x|) ≈ 1/24 divisor. EXP2_DOWNSCALE
// evaluates exp(|x|)/2^EXP2_DOWNSCALE (folded into exp's bias constant on the
// BF16 path, one exact integer add on the FP32 path's exponent), and
// _bessel_asymptotic_ multiplies the matching 2^EXP2_DOWNSCALE into P's
// coefficients, so callers pass them unscaled. With EXP2_DOWNSCALE=32 the exp
// intermediate peaks at 2.1e30 for |x|=92 instead of 9.0e39, and the only
// operation that can still overflow is the final rescaled multiply — which is
// where i_n itself leaves FP32, so overflowing there is the correct answer.
// ======================================================================

// 1/sqrt(x) via Quake-style magic constant + two Newton refinements
// (23-bit variant). Uses only literal constants — it does not touch
// vConstFloatPrgm*, so it is safe to call from a kernel whose init runs
// sfpu_reciprocal_init, which on Wormhole writes all three of them.
sfpi_inline sfpi::vFloat _rsqrt_quake_newton_23b_(const sfpi::vFloat x) {
    const sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - i);
    sfpi::vFloat c = (-y) * (x * y);
    y = y * (sfpi::vFloat(2.2825186f) + c * (sfpi::vFloat(2.2533049f) + c));
    c = 1.0f + (-y) * (x * y);
    return c * sfpi::addexp(y, -1) + y;
}

// The |x| callers clamp to before _bessel_asymptotic_, which static_asserts it
// against EXP2_DOWNSCALE. i0 (91.90076) and i1 (91.90626) both leave FP32
// below it, so a clamped input still overflows to +/-Inf at the final multiply.
constexpr float BESSEL_MAX_ABS_X = 92.0f;

// Computes exp(|x|)/2^EXP2_DOWNSCALE · 1/sqrt(|x|) · 2^EXP2_DOWNSCALE·P(1/|x|),
// the correctly-scaled asymptotic i_n(|x|). Callers pass P's coefficients
// c0..c5 unscaled: the 2^EXP2_DOWNSCALE is multiplied into each c_k below (an
// exact power of two, so the emitted constants change but no mantissa does).
//
// Precondition: the unsafe exp variants skip their range guards, so the caller
// must bound |x| to keep the biased result exponent |x|/ln2 + 127 -
// EXP2_DOWNSCALE below 255, i.e. |x| < (128 + EXP2_DOWNSCALE)·ln2 (110.9 at
// the default of 32). The static_assert below checks it at BESSEL_MAX_ABS_X,
// which needs EXP2_DOWNSCALE >= 5.
//
// INP_FLOAT32 selects between the FP32-accurate and BF16 21-bit exp variants,
// matching the caller's dtype macro. The rsqrt and poly evaluation are
// dtype-independent; the caller narrows to bf16 at store time if needed.
template <bool IS_FP32_INPUT, int EXP2_DOWNSCALE = 32>
sfpi_inline sfpi::vFloat _bessel_asymptotic_(
    const sfpi::vFloat abs_x,
    const float c0,
    const float c1,
    const float c2,
    const float c3,
    const float c4,
    const float c5) {
    static_assert(EXP2_DOWNSCALE >= 0 && EXP2_DOWNSCALE < 64, "R below needs a shift in [0, 64)");
    static_assert(
        BESSEL_MAX_ABS_X * 1.442695f + 127 - EXP2_DOWNSCALE < 255,
        "exp(BESSEL_MAX_ABS_X) / 2^EXP2_DOWNSCALE leaves FP32 in the unsafe exp");
    // 2^EXP2_DOWNSCALE, exact in FP32 for this range. Applied to the caller's
    // coefficients here rather than at the call site so the rescale can never
    // disagree with the downscale it undoes.
    constexpr float R = static_cast<float>(1ull << EXP2_DOWNSCALE);

    sfpi::vFloat exp_abs;
    if constexpr (IS_FP32_INPUT) {
        exp_abs = _sfpu_exp_fp32_accurate_unsafe_<EXP2_DOWNSCALE>(abs_x);
    } else {
        exp_abs = _sfpu_exp_21f_bf16_unsafe_<true, EXP2_DOWNSCALE>(abs_x);
    }

    // 1/sqrt(|x|) first, then 1/|x| as its square — one reciprocal saved.
    const sfpi::vFloat rsqrt_y = _rsqrt_quake_newton_23b_(abs_x);
    const sfpi::vFloat inv_abs_x = rsqrt_y * rsqrt_y;

    // P(y) at full precision. Nothing here is outlined (SFPI cannot pass a
    // vFloat through a real call), so these ops share calculate_i1's LRegs;
    // they fit because calculate_i1_asymptotic_ is reached from a v_if after
    // the polynomial path's temporaries are dead.
    const sfpi::vFloat correction =
        PolynomialEvaluator::eval(inv_abs_x, R * c0, R * c1, R * c2, R * c3, R * c4, R * c5);

    return exp_abs * rsqrt_y * correction;
}

}  // namespace ckernel::sfpu
