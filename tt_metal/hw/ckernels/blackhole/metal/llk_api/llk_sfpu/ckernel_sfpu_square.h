// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_square() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = v * v;
        sfpi::dst_reg++;
    }
}

// Square<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>: square_tile / square_tile_init (compute_kernel_api.h).
// init() is the shared SFPU init only.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Square : SfpuUnaryOp<Square<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_square<APPROXIMATION_MODE, ITERATIONS>(); }
};

}  // namespace ckernel::sfpu
