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
// Shared machinery for i0 and i1 asymptotic paths.
//
// Both compute the same shape past |x| > 10:
//   i_n(|x|) ≈ exp(|x|) / sqrt(|x|) · P(1/|x|)
// where P is a degree-5 minimax fit specific to the order n (Q for i0, P for i1).
// The two kernels differ only in the coefficient set and, for i1, a final
// sign fix-up (i0 is even, i1 is odd).
//
// exp(|x|) leaves FP32 at 88.72284 but i0/i1 do not until ≈91.90 — the
// asymptotic value carries a 1/sqrt(2·pi·|x|) ≈ 1/24 divisor. EXP2_DOWNSCALE
// evaluates exp(|x|)/2^EXP2_DOWNSCALE (folded into exp's own bias constant, so
// it costs nothing and rounds nothing), the matching 2^EXP2_DOWNSCALE is folded
// into P's coefficients by the caller. With EXP2_DOWNSCALE=32 the exp
// intermediate peaks at 2.1e30 for |x|=92 instead of 9.0e39, and the only
// operation that can still overflow is the final rescaled multiply — which is
// where i_n itself leaves FP32, so overflowing there is the correct answer.
// ======================================================================

// 1/sqrt(x) via Quake-style magic constant + two Newton refinements
// (23-bit variant). Uses only literal constants — it does not touch
// vConstFloatPrgm*, so it is safe to call from a kernel whose init runs
// sfpu_reciprocal_init, which writes vConstFloatPrgm0. The other two stay
// free; i1 uses neither.
sfpi_inline sfpi::vFloat _rsqrt_quake_newton_23b_(const sfpi::vFloat x) {
    const sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vInt(0x5f1110a0) - i);
    sfpi::vFloat c = (-y) * (x * y);
    y = y * (sfpi::vFloat(2.2825186f) + c * (sfpi::vFloat(2.2533049f) + c));
    c = 1.0f + (-y) * (x * y);
    return c * sfpi::addexp(y, -1) + y;
}

// Computes exp(|x|)/2^EXP2_DOWNSCALE · 1/sqrt(|x|) · P(1/|x|).
// Callers fold the matching 2^EXP2_DOWNSCALE into each c_k (an exact power of
// two — the emitted constants change but no mantissa does), so the return
// value is the correctly-scaled asymptotic i_n(|x|).
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
    static_assert(EXP2_DOWNSCALE >= 0 && EXP2_DOWNSCALE < 64, "R below assumes a non-negative shift");
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

    // P(y) evaluated at full precision; this outlined function does not stress
    // the main loop's LRA, so the extra ops are safe.
    const sfpi::vFloat correction =
        PolynomialEvaluator::eval(inv_abs_x, R * c0, R * c1, R * c2, R * c3, R * c4, R * c5);

    return exp_abs * rsqrt_y * correction;
}

}  // namespace ckernel::sfpu
