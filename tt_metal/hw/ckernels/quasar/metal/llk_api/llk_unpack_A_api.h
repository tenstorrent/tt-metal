// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_common_api.h"

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

    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, TRANSPOSE_EN, IS_32b_DEST_EN>(
        operand_id, 1 /*num_tiles_per_unpack*/);
}

/**
 *
 * @brief Unpacks a single operand, unpacker0 is used
 *
 * @param operand: The input operand circular buffer
 * @param tile_index: The index in the input CB to read from
 *
 * This function unpacks a single operand from the input circular buffer to srcA/dest register.
 */
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);
    // Use fifo_rd_tile_idx: number of tiles the read pointer has advanced from CB base
    const std::uint32_t l1_tile_index = get_local_cb_interface(operand_id).fifo_rd_tile_idx + tile_index;

    WAYPOINT("UPAW");
    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(l1_tile_index);
    WAYPOINT("UPAD");
}
