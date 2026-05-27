// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_common_api.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

/**
 *
 * @brief Initialize unpacker0 with dest reuse support
 *
 * @tparam IS_32b_DEST_EN: Enable using Math destination Register in 32-bit mode
 * @tparam BType, acc_to_dest, binary_reuse_dest, unpack_to_dest used to overload
 * matching Blackhole/Wormhole API signature to support binary dest reuse operations.
 * @param operand: The input operand circular buffer
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    [[maybe_unused]] const std::uint32_t transpose_of_faces = 0,
    [[maybe_unused]] const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);

    static_assert(unpack_to_dest == false, "unpack_to_dest is not yet supported on Quasar");
    static_assert(acc_to_dest == false, "acc_to_dest is not yet supported on Quasar");
    static_assert(BType == BroadcastType::NONE, "Only BroadcastType::NONE is supported on Quasar right now");

    // TODO (tt-metal #42916): Once runtime asserts are added, add asserts for unsupported features above and for valid
    // transpose_of_faces and within_face_16x16_transpose values

    // For Quasar, the unp_sel field is ignored if binary_reuse_dest != EltwiseBinaryReuseDestType::NONE
    if (transpose_of_faces && within_face_16x16_transpose) { /* TRANSPOSE_EN */
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, true /* TRANSPOSE_EN */, DST_ACCUM_MODE, binary_reuse_dest>(
            operand_id);
    } else {
        LLK_ASSERT(
            transpose_of_faces == within_face_16x16_transpose,
            "Quasar unpack unary operand only supports full or no transpose.");
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /* TRANSPOSE_EN */, DST_ACCUM_MODE, binary_reuse_dest>(
            operand_id);
    }
}

/**
 *
 * @brief Unpacks a single operand with dest reuse support
 *
 * @tparam BType, acc_to_dest, binary_reuse_dest, unpack_to_dest used to overload
 * matching Blackhole/Wormhole API signature to support binary dest reuse operations.
 * @param operand: The logical dataflow buffer id
 * @param tile_index: The index in the input CB to read from
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    const std::uint32_t l1_tile_index =
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx + tile_index;

    static_assert(unpack_to_dest == false, "unpack_to_dest is not yet supported on Quasar");
    static_assert(acc_to_dest == false, "acc_to_dest is not yet supported on Quasar");
    static_assert(BType == BroadcastType::NONE, "Only BroadcastType::NONE is supported on Quasar right now");

    WAYPOINT("UPAW");
    // For Quasar, the unp_sel field is ignored if binary_reuse_dest != EltwiseBinaryReuseDestType::NONE
    _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest>(l1_tile_index);
    WAYPOINT("UPAD");
}

/**
 * @brief Unpacks a contiguous block of tiles with unpacker0.
 *
 * @param operand The logical dataflow buffer id.
 * @param start_tile_index The starting tile index within the input buffer.
 * @param ntiles The number of consecutive tiles to unpack.
 *
 * The tiles are read from the operand buffer starting at start_tile_index
 * and unpacked into srcA one tile at a time.
 */
// TODO: AM; Optimize block calls by using ntiles per unpack, issue #40798
inline void llk_unpack_A_block(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    std::uint32_t l1_tile_index =
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx + start_tile_index;

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        _llk_unpack_unary_operand_<p_unpacr::UNP_A>(l1_tile_index);
        l1_tile_index += 1;
        WAYPOINT("UPAD");
    }
}
