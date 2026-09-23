// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardshrink(uint32_t param0) {
    // Hardshrink(x, λ) = x if |x| > λ, else 0
    // Single comparison using abs: setsgn(v, 0) clears sign bit
    // param0 contains lambda as FP32 bits. For BF16 inputs, the host pre-rounds
    // lambda to BF16 precision (then re-expands to FP32) so that FP32→FP19b
    // truncation on SFPU preserves the BF16 value exactly. For FP32 inputs,
    // lambda is passed as full FP32, so both input and lambda undergo the same
    // FP32→FP19b truncation, keeping comparisons consistent.
    sfpi::vFloat lambda = Converter::as_float(param0);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat abs_v = sfpi::setsgn(v, 0);
        v_if(abs_v <= lambda) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// Hardshrink<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode, lambda)
//   backs hardshrink_tile / hardshrink_tile_init (bare per-op init).
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Hardshrink : SfpuUnaryOp<Hardshrink<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(uint32_t param0) { calculate_hardshrink<APPROXIMATION_MODE, ITERATIONS>(param0); }
};
}  // namespace ckernel::sfpu
