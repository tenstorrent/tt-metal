// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

// Hardtanh(x) = max_val if x > max_val, min_val if x < min_val, else x
// Equivalent to: clamp(x, min_val, max_val) = min(max(x, min_val), max_val)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardtanh(uint param0, uint param1) {
    // Materialize both bounds outside the loop for better performance
    sfpi::vFloat min_val = Converter::as_float(param0);
    sfpi::vFloat max_val = Converter::as_float(param1);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::clamp(v, min_val, max_val);
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// Hardtanh<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode, min_val, max_val)
//   backs hardtanh_tile / hardtanh_tile_pack and their inits (bare per-op init).
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Hardtanh : SfpuUnaryOp<Hardtanh<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(uint32_t param0, uint32_t param1) {
        calculate_hardtanh<APPROXIMATION_MODE, ITERATIONS>(param0, param1);
    }
};
}  // namespace ckernel::sfpu
