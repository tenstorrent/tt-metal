// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
#include "sfpu/ckernel_sfpu_fill.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------------------------------
// Fill<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode, value) -> _calculate_fill_ (fill_tile, mul_reduce_scalar_tile)
//   init()                                   -> bare init        (fill_tile_init)
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Fill : SfpuUnaryOp<Fill<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(float value) { _calculate_fill_<APPROXIMATION_MODE, ITERATIONS>(value); }
};

// ---------------------------------------------------------------------------------------------------
// FillInt<APPROX, INSTRUCTION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode, value) -> _calculate_fill_int_ (fill_tile_int)
//   init()                                   -> bare init            (fill_tile_init)
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    InstrModLoadStore INSTRUCTION_MODE,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = 8>
struct FillInt
    : SfpuUnaryOp<FillInt<APPROXIMATION_MODE, INSTRUCTION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(std::uint32_t value) {
        _calculate_fill_int_<APPROXIMATION_MODE, INSTRUCTION_MODE, ITERATIONS>(value);
    }
};

// ---------------------------------------------------------------------------------------------------
// FillBitcast<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode, value_bit_mask) -> _calculate_fill_bitcast_ (fill_tile_bitcast)
//   init()                                            -> bare init                (fill_tile_init)
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct FillBitcast
    : SfpuUnaryOp<FillBitcast<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(std::uint32_t value_bit_mask) {
        _calculate_fill_bitcast_<APPROXIMATION_MODE, ITERATIONS>(value_bit_mask);
    }
};

}  // namespace sfpu
}  // namespace ckernel
