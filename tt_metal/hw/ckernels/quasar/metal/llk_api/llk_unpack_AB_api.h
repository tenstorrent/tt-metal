// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
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
 *
 * Each operand gets a BFD id allocated from the unpack partition (operandA on Unp0 / UNPACR0,
 * operandB on Unp1 / UNPACR1) and its table entry is programmed here; the DFB ids are used only
 * to fetch buffer info, never as BFD ids. One init burns 2 unpack-partition ids, so the partition
 * wraps sooner under mixed workloads — the standard wrap contract (re-init before re-execute)
 * applies.
 *
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

    // Neither LLK below takes a TensorShape, so neither scales its L1 tile index by the face count
    // (a tiny tile is registered as one HW tile per face, so they would move only a fraction of the
    // tile and step L1 by a single face). Full-tile only until they are converted (tt-metal #47597).
    LLK_ASSERT(
        get_operand_tensor_shape(operandA_id).total_num_faces() == ckernel::MAX_NUM_FACES,
        "this path indexes L1 in whole tiles, so it supports full 32x32 tiles only");
    LLK_ASSERT(
        get_operand_tensor_shape(operandB_id).total_num_faces() == ckernel::MAX_NUM_FACES,
        "this path indexes L1 in whole tiles, so it supports full 32x32 tiles only");

    llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp0>(operandA_id);
    llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp1>(operandB_id);

    if constexpr (BType == BroadcastType::NONE) {
        _llk_unpack_binary_operands_init_(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(),
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp1>(),
            1);
    } else {
        _llk_unpack_binary_broadcast_operands_init_<BType>(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(),
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp1>(),
            1);
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
    LLK_TDMA_GUARD_NOTE_TDMA(operandA);  // TEN-4746: real unpack (UNPACR) disarms these dfbs
    LLK_TDMA_GUARD_NOTE_TDMA(operandB);
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
