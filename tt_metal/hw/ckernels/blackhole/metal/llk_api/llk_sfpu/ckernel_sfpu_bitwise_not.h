// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitwise_not() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = ~v;
        dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// BitwiseNot<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode) -> calculate_bitwise_not (bitwise_not_tile)
//   init()                            -> bare init             (bitwise_not_tile_init)
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct BitwiseNot : SfpuUnaryOp<BitwiseNot<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_bitwise_not<APPROXIMATION_MODE, ITERATIONS>(); }
};
}  // namespace sfpu
}  // namespace ckernel
