// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = -val;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_int_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = -val;
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// Negative<APPROX, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode) -> _calculate_negative_ (float) / _calculate_negative_int_ (Int32)
//                                        (negative_tile, negative_tile_int32)
//   init()                            -> bare init                      (negative_tile_init)
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DataFormat FORMAT, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Negative
    : SfpuUnaryOp<Negative<APPROXIMATION_MODE, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() {
        if constexpr (FORMAT == DataFormat::Int32) {
            _calculate_negative_int_<APPROXIMATION_MODE, ITERATIONS>();
        } else {
            _calculate_negative_<APPROXIMATION_MODE, ITERATIONS>();
        }
    }
};
}  // namespace sfpu
}  // namespace ckernel
