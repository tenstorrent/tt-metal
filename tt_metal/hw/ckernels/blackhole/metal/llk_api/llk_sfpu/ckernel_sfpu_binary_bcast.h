// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_eltwise_binary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_binary_bcast.h"

namespace ckernel::sfpu {

// Op class for a binop between a data tile and the row or column broadcast of a second tile.
// The kernel walks Dest itself. BINOP is used only by run(): init() is shared by every BINOP.
template <BroadcastType BCAST_DIM, BinaryOp BINOP = BinaryOp::ADD>
struct BinaryBcast : SfpuBinaryOp<BinaryBcast<BCAST_DIM, BINOP>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate(
        const std::uint32_t dst_index_data, const std::uint32_t dst_index_bcast, const std::uint32_t dst_index_out) {
        _calculate_sfpu_binary_bcast_full_tile_<BINOP, BCAST_DIM>(dst_index_data, dst_index_bcast, dst_index_out);
    }

    static inline __attribute__((always_inline)) void init_op() { _sfpu_binary_bcast_init_<BCAST_DIM>(); }
};

}  // namespace ckernel::sfpu
