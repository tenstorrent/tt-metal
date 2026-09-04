// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tiled_prod() {
    vFloat result = 1.0f;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        result *= v;
        dst_reg[0] = result;
        dst_reg++;
    }
    vFloat v = dst_reg[0];
    result *= v;
    dst_reg[0] = result;
    dst_reg++;
}

// TiledProd<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>: tiled_prod_tile / tiled_prod_tile_init
// (compute_kernel_api.h). init() is the shared SFPU init only.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct TiledProd : SfpuUnaryOp<TiledProd<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_tiled_prod<APPROXIMATION_MODE, ITERATIONS>(); }
};

}  // namespace sfpu
}  // namespace ckernel
