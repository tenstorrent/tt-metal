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
 * @param full_ct_dim: Width of a full input in tiles
 TODO AM: Move full_ct_dim to runtime in tt-llk
 * @param c_dim_faces: Number of faces in c_dim per tile
 *
 * This function initializes unpack tilize for a tile row by full 32x32 tiles,
 * from the input circular buffer to srcA/srcB/dest register.
 *
 */
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN>
inline void llk_unpack_tilize_init(
    const std::uint32_t operand, const std::uint32_t full_ct_dim) {  // or should i pass full ct dim???
    const std::uint32_t operand_id = get_operand_id(operand);

    const c_dim_faces = (get_operand_narrow_tile(operand_id) ? 1 : 2);  // Tile width in faces

    // TODO: move ct_dim to runtime for unpack tilize __llk
    if (c_dim_faces == 2) {
        _llk_unpack_tilize_init_<UNP_SEL, IS_32b_DEST_EN, 2 /*c_dim_faces*/>(operand_id, full_ct_dim);
    } else {
        _llk_unpack_tilize_init_<UNP_SEL, IS_32b_DEST_EN, 1 /*c_dim_faces*/>(operand_id, full_ct_dim);
    }
}

/**
 *
 * @brief Performs unpack tilize on one 32x32 tile, using the selected unpacker resource
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = p_unpacr::UNP_A/UNP_B/UNP_DEST
 * @param operand: The input operand circular buffer identifier
 * @param tile_index: Index of the input tile in the input CB
 *
 * This function unpacks and tilizes one 32x32 tile, from the input circular buffer to srcA/srcB/dest
 * register.
 *
 */
template <std::uint32_t UNP_SEL>
inline void llk_unpack_tilize(std::uint32_t operand, std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);

    const std::uint32_t l1_tile_index = get_local_cb_interface(operand_id).fifo_rd_tile_idx + tile_index;

    WAYPOINT("UPTW");
    _llk_unpack_tilize_<UNP_SEL>(l1_tile_index);
    WAYPOINT("UPTD");
}

/**
 *
 * @brief Performs unpack tilize on a tile row by full 32x32 tiles, using the selected unpacker resource
 *
 * @tparam UNP_SEL: Selects which unpacker resource to use, values = p_unpacr::UNP_A/UNP_B/UNP_DEST
 * @param operand: The input operand circular buffer identifier
 TODO AM: Move full_ct_dim to runtime in tt-llk
 * @param full_ct_dim: Width of a full input in tiles
 * @param input_tile_index: Index of the input tile in the input CB
 *
 * This function unpacks and tilizes a tile row by full 32x32 tiles, from the input circular buffer to srcA/srcB/dest
 * register.
 *
 */
template <std::uint32_t UNP_SEL>
inline void llk_unpack_tilize_block(
    std::uint32_t operand, std::uint32_t full_ct_dim, std::uint32_t input_tile_index = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t narrow_tile = get_operand_narrow_tile(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t r_dim_faces = (num_faces == 2 && !narrow_tile) ? 1 : 2;  // Tile height in faces

    input_tile_index = input_tile_index * full_ct_dim * r_dim_faces * face_r_dim;
    for (std::uint32_t tile_index = 0; tile_index < full_ct_dim; tile_index++) {
        llk_unpack_tilize<UNP_SEL>(operand, input_tile_index + tile_index);
    }
}
