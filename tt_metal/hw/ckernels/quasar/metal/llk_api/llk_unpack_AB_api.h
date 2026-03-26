// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_binary_operands.h"
#include "llk_unpack_common_api.h"
#include "experimental/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK AB
 *************************************************************************/

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(
    const std::uint32_t operandA, const std::uint32_t operandB, [[maybe_unused]] const std::uint32_t transpose = 0) {
    static_assert(BType == BroadcastType::NONE, "Broadcast types will be added in a future update");

    // TODO: Once runtime asserts are added for Quasar, assert that transpose is unused
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    // num_tiles set to 1 for back-compatibility with existing APIs, can be increased in the future for better performance.
    _llk_unpack_binary_operands_init_(operandA_id, operandB_id, 1); 
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0) {

    static_assert(BType == BroadcastType::NONE, "Broadcast types will be added in a future update");
    // TODO: Once runtime asserts are added for Quasar, assert that bcast_row_idx is unused
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const std::uint32_t l1_tile_idx_a = g_dfb_interface[operandA_id].rd_entry_idx + tile_index_a;
    const std::uint32_t l1_tile_idx_b = g_dfb_interface[operandB_id].rd_entry_idx + tile_index_b;

    WAYPOINT("UABW");
    _llk_unpack_binary_operands_(l1_tile_idx_a, l1_tile_idx_b);
    WAYPOINT("UABD");
}