// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_log.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// erfinv(x) = x * p(u), u = ln(1 - x^2), after Giles, "Approximating the erfinv function" (GPU
// Computing Gems, 2010): a polynomial in u with one break at u = -7, so the central polynomial
// covers every bf16 and fp16 input (u >= -6.94) and the tail only fp32 inputs, where u reaches
// -15.9. Both are evaluated for every lane and selected.
//
// Unlike Giles, the tail is a polynomial in u itself rather than in sqrt(-u) - 3, so no square
// root is needed, and only the low-order coefficients are fp32 (an SFPLOADI pair each); the rest
// are fp16a- or bf16-exact and issue as one SFPLOADI. The coefficients were fitted with that
// rounding in the loop (round the top coefficient, refit the rest, repeat). Exact-arithmetic
// relative error: 5.7e-6 on the central and 1.3e-5 on the tail polynomial. The ln(1 - x^2)
// argument uses the log's fp32-accurate polynomial; with the bf16 one its 8.9e-4 absolute error
// dominated at 2.3e-4 relative. Winitzki's closed form, which this replaces, was ~2e-3 relative
// and returned 4.5e-4 for erfinv(0).
//
// Evaluated in u rather than -u: the negation is free (odd coefficients change sign) and the
// log's result feeds the polynomial directly. x = +/-1 gives u = -inf and a +inf tail
// polynomial; |x| > 1 gives a NaN log and a NaN result. The caller multiplies by x and copies
// its sign: SFPMUL returns +0 for a -0.0 input, and the sign of a NaN product follows the
// operands' signs through the chain, where the previous kernel matched x on both.
//
// Returns p(u) = erfinv(x) / x, not erfinv(x).
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat x) {
    // FAST_APPROX only skips the log's -0.0/subnormal normalisation (a * 1 + 0): 1 - x^2 comes out of
    // an SFPMAD, which flushes subnormals and returns +0.0 for the exact cancellation at x = +/-1, so
    // the normalisation changes no lane.
    sfpi::vFloat u = calculate_log_body<true, false, true>(1.0f - x * x, 0);

    // Central polynomial, u in [-7, 0], degree 7.
    sfpi::vFloat p = -0x1.3p-22f;
    p = p * u + -0x1.0ap-17f;
    p = p * u + -0x1.47cp-14f;
    p = p * u + -0x1.6b4p-13f;
    p = p * u + 0x1.3p-9f;
    p = p * u + 0x1.7b4fap-7f;
    p = p * u + -0x1.db1b3ap-3f;
    p = p * u + 0x1.c5c02p-1f;

    // Tail polynomial, u in [-16, -7], degree 4.
    sfpi::vFloat t = -0x1.aep-18f;
    t = t * u + -0x1.c24p-12f;
    t = t * u + -0x1.a58p-7f;
    t = t * u + -0x1.405c58p-2f;
    t = t * u + 0x1.93939cp-1f;

    v_if(u < -7.0f) { p = t; }
    v_endif;

    return p;
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() {
    constexpr int ITERATIONS = 8;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat p = calculate_erfinv_body<false>(sfpi::dst_reg[0]);
        // x is reloaded rather than kept live across the log, which needs every LREG.
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::copysgn(p * x, x);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erfinv_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    log_init<false, false, true>();
}

}  // namespace sfpu
}  // namespace ckernel
