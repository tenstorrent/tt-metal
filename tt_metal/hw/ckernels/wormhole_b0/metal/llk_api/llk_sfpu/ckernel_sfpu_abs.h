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
inline void calculate_abs() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // Kept as raw TTI intrinsics on purpose: SFPABS computes int32 abs in a single SFPU op. The sfpi
    // `v & 0x7FFFFFFF` form (#48598) needed two extra per-element SFPLOADI to rebuild the mask inside
    // the replay block, costing +30% cyc/tile.
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, InstrModLoadStore::INT32, 3, 0);
        TTI_SFPABS(0, 1, 0, 0);
        TTI_SFPSTORE(0, InstrModLoadStore::INT32, 3, 0);
        dst_reg++;
    }
}
// Abs<APPROX, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>: abs_tile (Float16_b), abs_tile_int32 (Int32) and
// abs_tile_init in compute_kernel_api.h. init() is the shared SFPU init only.
template <bool APPROXIMATION_MODE, DataFormat FORMAT, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Abs : SfpuUnaryOp<Abs<APPROXIMATION_MODE, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() {
        if constexpr (FORMAT == DataFormat::Int32) {
            calculate_abs_int32<APPROXIMATION_MODE, ITERATIONS>();
        } else {
            calculate_abs<APPROXIMATION_MODE, ITERATIONS>();
        }
    }
};

}  // namespace sfpu
}  // namespace ckernel
