// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// Reciprocal of two datums in lockstep. This is sfpu_reciprocal_iter<max_iter> with the two
// Newton-Raphson chains interleaved statement by statement: each chain is strictly serial
// (five SFPNOPs per datum on Wormhole) and the compiler does not interleave independent
// chains across rows on its own, so the second chain fills the first chain's stalls.
// The scale factor's sign is the complement of the input's (scale_bits = ~in), so the
// unsigned product y * scale is negative for these inputs (|v| + 1 >= 1 > 0) and
// sfpu_reciprocal_iter restores the sign with copysgn(y, in). Here the halving step
// multiplies by -0.5 instead of 0.5, which makes the scale positive at the same cost,
// and the copysgn (one SFPSETSGN per row) is dropped.
template <int max_iter>
sfpi_inline void _softsign_reciprocal_x2_(sfpi::vFloat& y0, sfpi::vFloat& y1, sfpi::vFloat in0, sfpi::vFloat in1) {
    sfpi::vFloat negative_x0 = sfpi::copyman(-1.0f, in0);
    sfpi::vFloat negative_x1 = sfpi::copyman(-1.0f, in1);
    y0 = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x0;
    y1 = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x1;
    sfpi::vUInt scale_bits0 = ~sfpi::as<sfpi::vUInt>(in0);
    sfpi::vUInt scale_bits1 = ~sfpi::as<sfpi::vUInt>(in1);
    y0 = sfpi::vConstFloatPrgm2 + y0 * negative_x0;
    y1 = sfpi::vConstFloatPrgm2 + y1 * negative_x1;
    sfpi::vFloat scale0 = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits0), 0);
    sfpi::vFloat scale1 = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits1), 0);
    sfpi::vFloat t0 = 1.0f + negative_x0 * y0;
    sfpi::vFloat t1 = 1.0f + negative_x1 * y1;
    scale0 *= -0.5f;
    scale1 *= -0.5f;
    y0 = y0 + y0 * t0;
    y1 = y1 + y1 * t1;
    if constexpr (max_iter > 1) {
        t0 = 1.0f + negative_x0 * y0;
        t1 = 1.0f + negative_x1 * y1;
        y0 = y0 + y0 * t0;
        y1 = y1 + y1 * t1;
    }
    y0 = y0 * scale0;
    y1 = y1 * scale1;
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
        _softsign_reciprocal_x2_<APPROXIMATION_MODE ? 0 : 2>(reciprocal0, reciprocal1, denominator0, denominator1);
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
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
