// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_binary_broadcast_operands.h"
#include "llk_unpack_binary_operands.h"
#include "llk_unpack_common_api.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK AB
 *************************************************************************/

/**
 * @brief Initialization for unpack of binary operations, uses SrcA & SrcB
 * @details Sets up MOP for unpacking binary operands
 * operandA will be used for UNPACKER0 -> SRCA
 * operandB will be used for UNPACKER1 -> SRCB
 * @tparam BType: Broadcast type for SrcB; one of {NONE, ROW, COL, SCALAR}.
 * @param operandA: The input operand dataflow buffer for source A
 * @param operandB: The input operand dataflow buffer for source B
 * @param transpose: Unused param; only for API compatibility.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(
    const std::uint32_t operandA, const std::uint32_t operandB, [[maybe_unused]] const ckernel::Transpose transpose) {
    // TODO (tt-metal #42916): Once runtime asserts are added for Quasar, assert that transpose is unused
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    if constexpr (BType == BroadcastType::NONE) {
        _llk_unpack_binary_operands_init_(operandA_id, operandB_id, 1);
    } else {
        _llk_unpack_binary_broadcast_operands_init_<BType>(operandA_id, operandB_id, 1);
    }
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    llk_unpack_AB_init<BType>(operandA, operandB, ckernel::Transpose::None);
}

/**
 * @brief Unpacks binary operands for SrcA & SrcB
 * @tparam BType: Broadcast type for SrcB; one of {NONE, ROW, COL, SCALAR}.
 * @param operandA: The logical dataflow buffer id for source A. Used to derive L1 addresses for SrcA unpacking.
 * @param operandB: The logical dataflow buffer id for source B. Used to derive L1 addresses for SrcB unpacking.
 * @param tile_index_a: Tile index within the operandA dataflow buffer to read from
 * @param tile_index_b: Tile index within the operandB dataflow buffer to read from
 * @param bcast_row_idx: Present for API compatibility with Blackhole binary `llk_unpack_AB` (ROW uses it there).
 *     Unused on Quasar; row selection within the B tile is not implemented in this wrapper.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0) {
    // TODO (tt-metal #42916): Once runtime asserts are added for Quasar, assert that bcast_row_idx is unused
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const LocalDFBInterface& local_dfb_interface_a = get_local_dfb_interface(operandA_id);
    const LocalDFBInterface& local_dfb_interface_b = get_local_dfb_interface(operandB_id);

    const std::uint32_t l1_tile_idx_a =
        local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].rd_entry_idx + tile_index_a;
    const std::uint32_t l1_tile_idx_b =
        local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].rd_entry_idx + tile_index_b;

    WAYPOINT("UABW");
    if constexpr (BType == BroadcastType::NONE) {
        _llk_unpack_binary_operands_(l1_tile_idx_a, l1_tile_idx_b);
    } else {
        _llk_unpack_binary_broadcast_operands_(l1_tile_idx_a, l1_tile_idx_b);
    }
    WAYPOINT("UABD");
}
