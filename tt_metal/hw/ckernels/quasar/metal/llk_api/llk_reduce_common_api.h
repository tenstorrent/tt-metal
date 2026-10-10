// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_operands.h"

/**
 * @brief Whether a reduce runs on the 2x-packed src-register format (MxFp4_2x_B).
 *
 * @tparam pool_type: Type of reduce pool op, values = [MAX, SUM, AVG]
 * @tparam reduce_dim: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @param operandA_id: The data operand id (SrcA)
 * @param operandB_id: The scaler operand id (SrcB)
 *
 * Column reduce is a GAPOOL (op-mmul family) and can consume MxFp4 as the 2x-packed src-register
 * format, like matmul, but only when the scaler is MxFp4 too: the 2x multiply is FP4 by FP4, and once
 * SrcA is 2x the FPU reads SrcB as packed FP4 pairs whatever SrcB holds, so a Float16_b scaler would be
 * misread. Only REDUCE_COL supports 2x (row/scalar reduce post-pool ELWADDDI does not), and only GAPOOL
 * (SUM/AVG) accepts a 2x SrcA - GMPOOL (MAX) does not. The single source of truth for both
 * @ref llk_unpack_AB_reduce_init (unpacker OUT_DATA_FORMAT) and @ref llk_math_reduce_init (ALU formats),
 * which must agree.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline bool is_2x_column_reduce(const std::uint32_t operandA_id, const std::uint32_t operandB_id) {
    if constexpr ((pool_type == PoolType::SUM || pool_type == PoolType::AVG) && reduce_dim == ReduceDim::REDUCE_COL) {
        return (static_cast<DataFormat>(get_operand_src_format(operandA_id)) == DataFormat::MxFp4) &&
               (static_cast<DataFormat>(get_operand_src_format(operandB_id)) == DataFormat::MxFp4);
    } else {
        return false;
    }
}
