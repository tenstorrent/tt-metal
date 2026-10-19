// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "llk_defs.h"
#include "tensor_shape.h"

namespace ckernel
{

// Tiles per unpack context: the data unpacker's Z counter addresses 256 faces.
constexpr std::uint32_t REDUCE_BLOCK_MAX_TILES = 32;

// Unpack and math agree on this: SUM/AVG REDUCE_ROW of full 32x32 tiles keeps the scaler in SrcA for a whole chunk.
// Not for BFP data: each tile starts with its exponent section, so the data unpacker cannot step to the next tile.
template <PoolType pool_type, ReduceDim reduce_dim>
constexpr bool reduce_block_holds_scaler(const TensorShape& tensor_shape, const std::uint32_t data_format)
{
    return (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX) && (tensor_shape.total_num_faces() == 4) &&
           (tensor_shape.face_r_dim == FACE_R_DIM) && !IS_BFP_FORMAT(data_format);
}

}
