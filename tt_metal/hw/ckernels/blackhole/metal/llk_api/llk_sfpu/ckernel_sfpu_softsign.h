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
// The reciprocal is sfpu_reciprocal_iter<max_iter> from ckernel_sfpu_recip.h (same seed and
// Newton-Raphson steps, same NaN check; keep the two in sync) with the two rows' chains interleaved
// statement by statement. Each chain is strictly serial and the compiler does not interleave
// independent rows on its own, so the second chain fills the first chain's dependency stalls.
template <int max_iter>
sfpi_inline void _softsign_x2_(sfpi::vFloat& out0, sfpi::vFloat& out1, sfpi::vFloat in0, sfpi::vFloat in1) {
    sfpi::vFloat y0 = sfpi::approx_recip(in0);
    sfpi::vFloat y1 = sfpi::approx_recip(in1);
    if constexpr (max_iter > 0) {
        // t = -(2 - in*y); t >= 0 (NaN included) marks the exceptional in = 0 / inf lanes.
        sfpi::vFloat t0 = in0 * y0 - sfpi::vConstFloatPrgm0;
        sfpi::vFloat t1 = in1 * y1 - sfpi::vConstFloatPrgm0;
        sfpi::vFloat y0_1 = y0 * -t0 - 0.0f;
        sfpi::vFloat y1_1 = y1 * -t1 - 0.0f;
        v_if(t0 < 0) {
            t0 = in0 * y0_1 - sfpi::vConstFloatPrgm0;
            y0 = y0_1 * -t0 - 0.0f;
        }
        v_endif;
        v_if(t1 < 0) {
            t1 = in1 * y1_1 - sfpi::vConstFloatPrgm0;
            y1 = y1_1 * -t1 - 0.0f;
        }
        v_endif;
    }
    out0 = y0;
    out1 = y1;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
    static_assert(ITERATIONS % 2 == 0, "calculate_softsign processes two rows per iteration");
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
        sfpi::dst_reg[0] = v0 * reciprocal0;
        sfpi::dst_reg[1] = v1 * reciprocal1;
        sfpi::dst_reg += 2;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
