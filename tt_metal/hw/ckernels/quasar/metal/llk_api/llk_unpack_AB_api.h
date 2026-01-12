// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_binary_operands.h"
#include "llk_unpack_binary_broadcast_operands.h"

/*************************************************************************
 * LLK UNPACK AB
 *************************************************************************/

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    if constexpr (BType == BroadcastType::NONE) {
        _llk_unpack_binary_operands_init_(operandA_id, operandB_id, 1 /*num_tiles_per_unpack*/);
    } else {
        _llk_unpack_binary_broadcast_operands_init_<BType>(operandA_id, operandB_id, 1 /*num_tiles_per_unpack*/);
    }
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);

    std::uint32_t l1_tile_index_a = get_local_cb_interface(operandA_id).fifo_rd_tile_idx + tile_index_a;
    std::uint32_t l1_tile_index_b = get_local_cb_interface(operandB_id).fifo_rd_tile_idx + tile_index_b;

    WAYPOINT("UABW");
    if constexpr (BType == BroadcastType::NONE) {
        _llk_unpack_binary_operands_(l1_tile_index_a, l1_tile_index_b);
    } else {
        _llk_unpack_binary_broadcast_operands_(l1_tile_index_a, l1_tile_index_b);
    }
    WAYPOINT("UABD");
}
