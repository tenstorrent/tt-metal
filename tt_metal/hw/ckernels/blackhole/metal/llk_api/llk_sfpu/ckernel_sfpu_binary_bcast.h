// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "llk_math_eltwise_sfpu_op.h"
#include "sfpu/ckernel_sfpu_binary_bcast.h"

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------------------------------
// BinaryBcast<BINOP, BCAST_DIMENSION, DST_SYNC, DST_ACCUM>
//   (the parameter is deliberately not named BCAST_DIM: bcast kernels #define BCAST_DIM before including the API)
//   calculate(data, bcast, out, VectorMode::None) -> _calculate_sfpu_binary_bcast_full_tile_<BINOP, BCAST_DIMENSION>
//   init()                                        -> _sfpu_binary_bcast_init_<BCAST_DIMENSION> (BINOP-agnostic)
//   Backs sfpu_{add,sub,mul}_bcast_{col,row}(_init), sfpu_bcast_{col,row}_init, sfpu_bcast(_init)
//   (api/compute/sfpu_binary_bcast.h).
// ---------------------------------------------------------------------------------------------------
template <BinaryOp BINOP, BroadcastType BCAST_DIMENSION, DstSync DST_SYNC, bool DST_ACCUM>
struct BinaryBcast : SfpuBinaryOp<BinaryBcast<BINOP, BCAST_DIMENSION, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM> {
    static void kernel(std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out) {
        _calculate_sfpu_binary_bcast_full_tile_<BINOP, BCAST_DIMENSION>(dst_index_data, dst_index_bcast, dst_index_out);
    }

    static void init_kernel() { _sfpu_binary_bcast_init_<BCAST_DIMENSION>(); }
};

}  // namespace sfpu
}  // namespace ckernel
