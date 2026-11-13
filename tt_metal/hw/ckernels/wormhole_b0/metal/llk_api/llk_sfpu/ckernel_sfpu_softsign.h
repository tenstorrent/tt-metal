// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// Reciprocal of two softsign denominators in lockstep: out = 1 / in for in = |v| + 1.
//
// The reciprocal is sfpu_reciprocal_iter<max_iter> from ckernel_sfpu_recip.h (same seed,
// scale and Newton-Raphson steps; keep the two in sync) with the two rows' chains interleaved
// statement by statement. Each chain is strictly serial: one row at a time, the loop issues
// six SFPNOPs per row with max_iter == 2 and four with max_iter == 0, and the compiler does
// not interleave independent rows on its own, so the second chain fills the first chain's
// stalls.
//
// Precondition: in > 0, which |v| + 1 satisfies for every input except -NaN. The chain runs
// on sfpu_reciprocal_iter's values multiplied by -1/2, which drops its scale halving and its
// copysgn (one SFPMULI and one SFPSETSGN per row):
// - For in > 0 the scale setman(~in, 0) is -2 * 2^-e (e = in's unbiased exponent), and
//   sfpu_reciprocal_iter halves it and restores the sign with copysgn(y, in). Here the
//   normalised input is x = setexp(in, 128) = 2 * mantissa instead of copyman(-1.0, in) =
//   -mantissa (one SFPSETEXP instead of an SFPMOV and an SFPSETMAN), and
//   init_softsign programs the seed k0, k1, k2 as -k0/8, k1/4, -k2/2, so every seed and
//   Newton-Raphson value is exactly -1/2 times sfpu_reciprocal_iter's (t is the same value)
//   and y * scale is the same product, rounded and flushed the same way: 0 for |v| in
//   (2^126, 2^127) where 1/in is below the smallest normal.
// - -NaN: Wormhole sfpi::abs keeps a NaN's sign, so in is -NaN and the precondition fails. The
//   result keeps the input's sign (-NaN, or -inf through a 16-bit Dest), where the copysgn(y, in)
//   of sfpu_reciprocal_iter makes it positive.
// Returns 1 / in per row; the caller multiplies by v. Peak: scale, x, y and t per row, which
// fill the 8 LRegs for two rows.
template <int max_iter>
sfpi_inline void _softsign_x2_(sfpi::vFloat& out0, sfpi::vFloat& out1, sfpi::vFloat in0, sfpi::vFloat in1) {
    constexpr int BIASED_EXPONENT_OF_TWO = 128;
    sfpi::vFloat x0 = sfpi::setexp(in0, BIASED_EXPONENT_OF_TWO);
    sfpi::vFloat x1 = sfpi::setexp(in1, BIASED_EXPONENT_OF_TWO);
    sfpi::vUInt scale_bits0 = ~sfpi::as<sfpi::vUInt>(in0);
    sfpi::vUInt scale_bits1 = ~sfpi::as<sfpi::vUInt>(in1);
    sfpi::vFloat scale0 = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits0), 0);
    sfpi::vFloat scale1 = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits1), 0);
    sfpi::vFloat y0 = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * x0;
    sfpi::vFloat y1 = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * x1;
    y0 = sfpi::vConstFloatPrgm2 + y0 * x0;
    y1 = sfpi::vConstFloatPrgm2 + y1 * x1;
    sfpi::vFloat t0 = 1.0f + x0 * y0;
    sfpi::vFloat t1 = 1.0f + x1 * y1;
    y0 = y0 + y0 * t0;
    y1 = y1 + y1 * t1;
    if constexpr (max_iter > 1) {
        t0 = 1.0f + x0 * y0;
        t1 = 1.0f + x1 * y1;
        y0 = y0 + y0 * t0;
        y1 = y1 + y1 * t1;
    }
    out0 = y0 * scale0;
    out1 = y1 * scale1;
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
        sfpi::vFloat reciprocal0, reciprocal1;
        _softsign_x2_<APPROXIMATION_MODE ? 0 : 2 /*max_iter*/>(reciprocal0, reciprocal1, denominator0, denominator1);
        // Reload v: keeping both inputs live across the two interleaved chains exceeds the 8 LRegs.
        v0 = sfpi::dst_reg[0];
        v1 = sfpi::dst_reg[1];
        sfpi::vFloat out0 = v0 * reciprocal0;
        sfpi::vFloat out1 = v1 * reciprocal1;
        sfpi::dst_reg[0] = out0;
        sfpi::dst_reg[1] = out1;
        sfpi::dst_reg += 2;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // sfpu_reciprocal_init's shared seed, rescaled for _softsign_x2_'s x = 2 * mantissa and
    // y = -1/x: k2' + (k1' + k0'*x)*x = -(k2 - k1*(x/2) + k0*(x/2)**2) / 2.
    sfpi::vConstFloatPrgm0 = -RECIPROCAL_SEED_K0 / 8;
    sfpi::vConstFloatPrgm1 = RECIPROCAL_SEED_K1 / 4;
    sfpi::vConstFloatPrgm2 = -RECIPROCAL_SEED_K2 / 2;
}

}  // namespace ckernel::sfpu
