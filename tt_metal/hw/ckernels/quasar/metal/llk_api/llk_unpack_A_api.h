// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_unary_operand_api.h"
#include "llk_unpack_unary_broadcast_operands_api.h"

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
    llk_unpack_unary_operand_init<p_unpacr::UNP_A, TRANSPOSE_EN, IS_32b_DEST_EN>(operand);
}

/**
 * Shared compute-kernel entry point (same template surface as Blackhole / other metal arches):
 * unary and unary-broadcast unpack for one operand.
 * Quasar implements broadcast via llk_unpack_unary_broadcast_operands_* (not _llk_unpack_A_*).
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(const std::uint32_t transpose_of_faces = 0, const std::uint32_t operand = 0) {
    constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
    if constexpr (BType == BroadcastType::NONE) {
        if (transpose_of_faces != 0) {
            llk_unpack_unary_operand_init<unp_sel, true, DST_ACCUM_MODE, binary_reuse_dest>(operand, 1);
        } else {
            llk_unpack_unary_operand_init<unp_sel, false, DST_ACCUM_MODE, binary_reuse_dest>(operand, 1);
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
    llk_unpack_unary_operand_tile<p_unpacr::UNP_A>(operand, tile_index);
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
        llk_unpack_unary_operand_tile<unp_sel, binary_reuse_dest>(operand, tile_index);
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
    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        llk_unpack_unary_operand_tile<p_unpacr::UNP_A>(operand, tile_index);
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
