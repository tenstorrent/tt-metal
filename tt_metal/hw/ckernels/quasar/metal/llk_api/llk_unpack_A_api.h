// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_unary_broadcast_operands_api.h"
#include "llk_unpack_unary_operand.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

/**
 *
 * @brief Initialize selected unpacker to unpack a single tile
 *
 * @tparam TRANSPOSE_EN: Enables transpose of a tile, supported for SrcA and SrcB
 * @tparam IS_32b_DEST_EN: Enable using Math destination Register in 32-bit mode
 * @param operand: The input operand circular buffer
 *
 * This function initializes unpacker0 to unpack a single tile
 * from the input circular buffer to srcA/dest register.
 */
template <bool TRANSPOSE_EN, bool IS_32b_DEST_EN>
inline void llk_unpack_A_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, TRANSPOSE_EN, IS_32b_DEST_EN>(operand_id, NUM_TILES, num_faces);
}

/**
 *
 * @brief Initialize unpacker for unary and unary-broadcast operations on one operand.
 *
 * @tparam BType: Broadcast type; BroadcastType::NONE selects the plain unary path
 * @tparam acc_to_dest: Unused on Quasar; kept for API parity with Blackhole / other arches
 * @tparam binary_reuse_dest: Dest reuse mode
 * @tparam unpack_to_dest: When true, unpack targets dest (UNP_A); otherwise SrcB (UNP_B)
 * @param transpose_of_faces: Non-zero enables transpose of 16x16 faces (unary path only)
 * @param operand: The input operand logical dataflow buffer / CB id
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(const std::uint32_t transpose_of_faces = 0, const std::uint32_t operand = 0) {
    constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
    if constexpr (BType == BroadcastType::NONE) {
        const std::uint32_t operand_id = get_operand_id(operand);
        const std::uint32_t num_faces = get_operand_num_faces(operand_id);
        if (transpose_of_faces != 0) {
            _llk_unpack_unary_operand_init_<unp_sel, true, DST_ACCUM_MODE, binary_reuse_dest>(operand_id, 1, num_faces);
        } else {
            _llk_unpack_unary_operand_init_<unp_sel, false, DST_ACCUM_MODE, binary_reuse_dest>(
                operand_id, 1, num_faces);
        }
    } else {
        constexpr bool is_fp32_dest_acc_en = unpack_to_dest ? false : DST_ACCUM_MODE;
        llk_unpack_unary_broadcast_operands_init<unp_sel, BType, unpack_to_dest, is_fp32_dest_acc_en>(operand, 1);
    }
}

/**
 *
 * @brief Unpacks a single operand, unpacker0 is used
 *
 * @param operand: The logical dataflow buffer id
 * @param tile_index: The index in the input CB to read from
 *
 * This function unpacks a single operand from the input circular buffer to srcA/dest register.
 */
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    WAYPOINT("UPAW");
    const std::uint32_t operand_id = get_operand_id(operand);
    const auto& local_dfb = g_dfb_interface[operand_id];
    const std::uint32_t l1_tile_idx = local_dfb.tc_slots[local_dfb.tc_idx].rd_entry_idx + tile_index;
    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(l1_tile_idx);
    WAYPOINT("UPAD");
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    WAYPOINT("UPAW");
    constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
    if constexpr (BType == BroadcastType::NONE) {
        const std::uint32_t operand_id = get_operand_id(operand);
        const auto& local_dfb = g_dfb_interface[operand_id];
        const std::uint32_t l1_tile_idx = local_dfb.tc_slots[local_dfb.tc_idx].rd_entry_idx + tile_index;
        _llk_unpack_unary_operand_<unp_sel, binary_reuse_dest>(l1_tile_idx);
    } else {
        llk_unpack_unary_broadcast_operands<unp_sel, unpack_to_dest>(operand, tile_index);
    }
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
    const auto& local_dfb = g_dfb_interface[operand_id];
    const std::uint32_t rd_entry_idx = local_dfb.tc_slots[local_dfb.tc_idx].rd_entry_idx;
    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        _llk_unpack_unary_operand_<p_unpacr::UNP_A>(rd_entry_idx + tile_index);
        WAYPOINT("UPAD");
    }
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_block(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {
    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        llk_unpack_A<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(operand, tile_index);
    }
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_A_uninit(const std::uint32_t /*operand*/) {}
