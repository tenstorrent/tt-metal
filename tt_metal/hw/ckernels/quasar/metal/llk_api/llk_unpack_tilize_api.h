// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_tilize.h"
#include "experimental/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK TILIZE
 *************************************************************************/

/**
 * @brief Initializes the unpacker for tilize operations on Quasar.
 *
 * Configures UNP_A stride registers and programs the MOP for tilizing
 * BLOCK_CT_DIM tiles from row-major L1 data into face format in SrcA.
 *
 * @tparam BLOCK_CT_DIM  Number of tiles per row (block width in tiles).
 * @param operand        The input dataflow buffer identifier.
 */
template <std::uint32_t BLOCK_CT_DIM>
inline void llk_unpack_tilize_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);

    // TODO: Once narrow-tile is supported c_dim_faces will be variable.
    constexpr std::uint32_t c_dim_faces = 2;

    _llk_unpack_tilize_init_<p_unpacr::UNP_A, DST_ACCUM_MODE, BLOCK_CT_DIM, BLOCK_CT_DIM, c_dim_faces>(operand_id);
}

/**
 * @brief Tilizes a block of tiles from L1 row-major layout into SrcA.
 *
 * Computes the L1 face index from the DFB read position and the input
 * tile index, then runs the MOP configured by llk_unpack_tilize_init.
 *
 * @param operand          The input dataflow buffer identifier.
 * @param block_c_tiles    Number of tiles in one block row (must match BLOCK_CT_DIM from init).
 * @param input_tile_index Starting tile index (encodes row offset via block_c_tiles stride).
 */
inline void llk_unpack_tilize_block(
    const std::uint32_t operand, const std::uint32_t block_c_tiles, const std::uint32_t input_tile_index = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);

    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);
    const std::uint32_t R_DIM_FACES = (num_faces == 2 && !narrow_tile) ? 1 : 2;
    const std::uint32_t faces_per_entry = R_DIM_FACES * face_r_dim;

    const LocalDFBInterface& local_dfb = g_dfb_interface[operand_id];
    const std::uint32_t rd_entry_idx = local_dfb.tc_slots[local_dfb.tc_idx].rd_entry_idx;

    // Determine which tile-row this index falls in
    const std::uint32_t row = input_tile_index / block_c_tiles;
    const std::uint32_t l1_face_idx = (rd_entry_idx + row * block_c_tiles) * faces_per_entry;

    _llk_unpack_tilize_<p_unpacr::UNP_A>(l1_face_idx);
}
