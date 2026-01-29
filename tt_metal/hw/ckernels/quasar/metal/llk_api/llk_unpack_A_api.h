// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = p_unpacr::UNP_A/UNP_B/UNP_DEST
 * @tparam TRANSPOSE_EN: Enables transpose of a tile, supported for SrcA and SrcB
 * @tparam IS_32b_DEST_EN: Enable using Math destination Register in 32-bit mode
 * @param operand: The input operand circular buffer
 *
 * This function initializes the selected unpacker to unpack a single tile
 * from the input circular buffer to srcA/srcB/dest register.
 */
template <std::uint32_t UNP_SEL, bool TRANSPOSE_EN, bool IS_32b_DEST_EN>
inline void llk_unpack_A_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);

    _llk_unpack_unary_operand_init_<UNP_SEL, TRANSPOSE_EN, IS_32b_DEST_EN>(operand_id, 1 /*num_tiles_per_unpack*/);
}

/**
 *
 * @brief Unpacks a single operand, works for any unpack resource
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = p_unpacr::UNP_A/UNP_B/UNP_DEST
 * @param operand: The input operand circular buffer
 * @param tile_index: The index in the input CB to read from
 *
 * This function unpacks a single operand from the input circular buffer to srcA/srcB/dest register.
 */
template <std::uint32_t UNP_SEL>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);
    // Use fifo_rd_tile_idx: number of tiles the read pointer has advanced from CB base
    const std::uint32_t l1_tile_index = get_local_cb_interface(operand_id).fifo_rd_tile_idx + tile_index;

    WAYPOINT("UPAW");
    _llk_unpack_unary_operand_<UNP_SEL>(l1_tile_index);
    WAYPOINT("UPAD");
}
