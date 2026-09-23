// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softshrink(uint32_t param0) {
    // Softshrink(x) = x - λ if x > λ, x + λ if x < -λ, else 0
    // SFPU microcode
    sfpi::vFloat lambda = Converter::as_float(param0);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 0.0f;
        v_if(v > lambda) { sfpi::dst_reg[0] = v - lambda; }
        v_elseif(v < (-lambda)) { sfpi::dst_reg[0] = v + lambda; }
        v_endif;
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// Softshrink<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode, lambda)
//   backs softshrink_tile / softshrink_tile_init (bare per-op init).
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Softshrink : SfpuUnaryOp<Softshrink<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(uint32_t param0) { calculate_softshrink<APPROXIMATION_MODE, ITERATIONS>(param0); }
};
}  // namespace ckernel::sfpu
