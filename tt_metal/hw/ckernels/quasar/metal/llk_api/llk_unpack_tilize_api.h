// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_tilize.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK TILIZE
 *************************************************************************/

/**
 *
 * @brief Initializes the selected unpacker to unpack tilize a tile row by full 32x32 tiles
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = p_unpacr::UNP_A/UNP_B/UNP_DEST
 * @tparam IS_32b_DEST_EN: Set to true to enable using math destination register in 32b mode
 * @param operand: The input operand circular buffer identifier
 * @param full_ct_dim: Number of tiles in a row of the input tensor. Input tensor is in row-major format.
 * @param block_ct_dim: c_dim of tiles in each block
 * @param c_dim_faces: Number of faces in c_dim per tile
 *
 * This function initializes unpack tilize for a tile row by full 32x32 tiles,
 * from the input circular buffer to srcA/srcB/dest register.
 *
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN>
inline void llk_unpack_tilize_init(
    const std::uint32_t operand,
    const std::uint32_t full_ct_dim,
    const std::uint32_t block_ct_dim,
    const std::uint32_t c_dim_faces) {
    const std::uint32_t operand_id = get_operand_id(operand);

    _llk_unpack_tilize_init_<UNP_SEL, IS_32b_DEST_EN>(operand_id, full_ct_dim, block_ct_dim, c_dim_faces);
}

/**
 *
 * @brief Performs unpack tilize on a tile row by full 32x32 tiles, using the selected unpacker resource
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = p_unpacr::UNP_A/UNP_B/UNP_DEST
 * @param operand: The input operand circular buffer identifier
 * @param tile_index: The L1 index in the input CB to read from
 *
 * This function unpacks and tilizes a tile row by full 32x32 tiles, from the input circular buffer to srcA/srcB/dest
 * register.
 *
 */
inline void llk_unpack_tilize(std::uint32_t operand, std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);

    const std::uint32_t l1_tile_index = get_local_cb_interface(operand_id).fifo_rd_tile_idx + tile_index;

    WAYPOINT("UPTW");
    _llk_unpack_tilize_<UNP_SEL>(l1_tile_index);
    WAYPOINT("UPTD");
}
