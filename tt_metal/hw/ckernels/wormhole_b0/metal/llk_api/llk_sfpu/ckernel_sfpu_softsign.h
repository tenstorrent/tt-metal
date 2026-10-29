// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// Softsign of two rows in lockstep: out = v * (1 / in) for in = |v| + 1.
//
// The reciprocal is sfpu_reciprocal_iter<max_iter> from ckernel_sfpu_recip.h (same seed
// constants, scale and Newton-Raphson steps; keep the two in sync) with the two rows' chains
// interleaved statement by statement. Each chain is strictly serial: one row at a time, the
// loop issues six SFPNOPs per row with max_iter == 2 and four with max_iter == 0, and the
// compiler does not interleave independent rows on its own, so the second chain fills the
// first chain's stalls.
//
// Precondition: in > 0, which |v| + 1 satisfies for every input except -NaN. Two departures
// from sfpu_reciprocal_iter rely on it:
// - The scale factor gets the complement of the input's sign (scale_bits = ~in), so for in > 0
//   it is negative and sfpu_reciprocal_iter restores the sign with copysgn(y, in). Here the
//   halving step multiplies by -0.5 instead of 0.5, which makes the scale positive at the same
//   cost, and the copysgn (one SFPSETSGN per row) is dropped.
// - The power-of-two scale multiplies v instead of the reciprocal: v * scale is exact, so
//   (v * scale) * y rounds once, like v * (y * scale), and needs no reload of v from Dest. For
//   |v| in [2^126, 2^127), y * scale is below the smallest normal and flushes to zero, so
//   v * (y * scale) would return 0; (v * scale) * y returns the correct value of about +-1.
// -NaN: Wormhole sfpi::abs keeps a NaN's sign, so in is -NaN and the precondition fails. The
// result keeps the input's sign (-NaN, or -inf through a 16-bit Dest), where the copysgn(y, in)
// of sfpu_reciprocal_iter would make it positive.
template <int max_iter>
sfpi_inline void _softsign_x2_(
    sfpi::vFloat& out0, sfpi::vFloat& out1, sfpi::vFloat v0, sfpi::vFloat v1, sfpi::vFloat in0, sfpi::vFloat in1) {
    sfpi::vFloat negative_x0 = sfpi::copyman(-1.0f, in0);
    sfpi::vFloat negative_x1 = sfpi::copyman(-1.0f, in1);
    // Scale before seed: v and in die before y and t are live, so the peak is scaled_v,
    // negative_x, y and t per row, which fills the 8 LRegs for two rows (seed first does not fit).
    sfpi::vUInt scale_bits0 = ~sfpi::as<sfpi::vUInt>(in0);
    sfpi::vUInt scale_bits1 = ~sfpi::as<sfpi::vUInt>(in1);
    sfpi::vFloat scale0 = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits0), 0);
    sfpi::vFloat scale1 = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits1), 0);
    scale0 *= -0.5f;
    scale1 *= -0.5f;
    sfpi::vFloat scaled_v0 = v0 * scale0;
    sfpi::vFloat scaled_v1 = v1 * scale1;
    sfpi::vFloat y0 = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x0;
    sfpi::vFloat y1 = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x1;
    y0 = sfpi::vConstFloatPrgm2 + y0 * negative_x0;
    y1 = sfpi::vConstFloatPrgm2 + y1 * negative_x1;
    sfpi::vFloat t0 = 1.0f + negative_x0 * y0;
    sfpi::vFloat t1 = 1.0f + negative_x1 * y1;
    y0 = y0 + y0 * t0;
    y1 = y1 + y1 * t1;
    if constexpr (max_iter > 1) {
        t0 = 1.0f + negative_x0 * y0;
        t1 = 1.0f + negative_x1 * y1;
        y0 = y0 + y0 * t0;
        y1 = y1 + y1 * t1;
    }
    out0 = scaled_v0 * y0;
    out1 = scaled_v1 * y1;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
    static_assert(ITERATIONS % 2 == 0, "calculate_softsign processes two rows per iteration");
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat denominator0 = sfpi::abs(v0) + 1.0f;
        sfpi::vFloat denominator1 = sfpi::abs(v1) + 1.0f;
        sfpi::vFloat out0, out1;
        _softsign_x2_<APPROXIMATION_MODE ? 0 : 2 /*max_iter*/>(out0, out1, v0, v1, denominator0, denominator1);
        sfpi::dst_reg[0] = out0;
        sfpi::dst_reg[1] = out1;
        sfpi::dst_reg += 2;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
