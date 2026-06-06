// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_cb_tile_access.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB REDUCE
 *************************************************************************/

template <PoolType pool_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
inline void llk_unpack_AB_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    if constexpr (enforce_fp32_accumulation) {
        // Set necessary config regs for MOVB2D hi16/lo16 to work
        _llk_unpack_dbg_feature_disable_();
    }

    _llk_unpack_AB_reduce_init_<pool_type, reduce_dim, enforce_fp32_accumulation>(tensor_shape);
}

template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t address_a = llk_unpack_tile_address(operandA_id, tile_index_a);
    std::uint32_t address_b = llk_unpack_tile_address(operandB_id, tile_index_b);

    LLK_ASSERT_BLOCK(validate_unpack_tile_access(operandA_id, tile_index_a, 1));
    LLK_ASSERT_BLOCK(validate_unpack_tile_access(operandB_id, tile_index_b, 1));

    WAYPOINT("UABW");
    _llk_unpack_AB_reduce_<pool_type, reduce_dim>(address_a, address_b);
    WAYPOINT("UABD");
}
