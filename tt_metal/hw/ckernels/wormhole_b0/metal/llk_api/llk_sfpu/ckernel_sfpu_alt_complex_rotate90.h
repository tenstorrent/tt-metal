// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_alt_complex_rotate90() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        dst_reg[0] = -vFloat(dst_reg[1]);
        dst_reg[1] = val;
        dst_reg += 2;
    }
}

// AltComplexRotate90<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>: alt_complex_rotate90_tile /
// alt_complex_rotate90_tile_init (compute_kernel_api.h). Uses the bare per-op init
//. The kernel processes 4 rows per iteration, hence ITERATIONS = 4.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 4>
struct AltComplexRotate90
    : SfpuUnaryOp<AltComplexRotate90<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_alt_complex_rotate90<APPROXIMATION_MODE, ITERATIONS>(); }
};

}  // namespace sfpu
}  // namespace ckernel
