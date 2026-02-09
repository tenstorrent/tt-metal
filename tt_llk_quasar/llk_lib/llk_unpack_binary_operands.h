// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @brief MOP configuration for unpack of binary operations, uses SrcA & SrcB
 * @details Sets up MOP for unpacking binary operands tile by tile
 * buf_desc_id_0 will be used for UNPACKER0 -> SRCA
 * buf_desc_id_1 will be used for UNPACKER1 -> SRCB
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for both inputs
 */
inline void _llk_unpack_binary_operands_mop_config_(const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t num_tiles)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = num_tiles;

    std::uint32_t unpack_instrn0 = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
    std::uint32_t unpack_instrn1 = TT_OP_UNPACR1_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_1, 1 /*Set Dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_instrn0, unpack_instrn1);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialization for unpack of binary operations, uses SrcA & SrcB
 * @details Sets up MOP for unpacking binary operands
 * buf_desc_id_0 will be used for UNPACKER0 -> SRCA
 * buf_desc_id_1 will be used for UNPACKER1 -> SRCB
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for both inputs
 */
inline void _llk_unpack_binary_operands_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t num_tiles = NUM_TILES_PER_UNPACK)
{
    _llk_unpack_binary_operands_mop_config_(buf_desc_id_0, buf_desc_id_1, num_tiles);
}

/**
 * @brief Unpacks binary operands for SrcA & SrcB
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer
 * start_l1_tile_idx_0 -> UNPACKER0 -> SRCA
 * start_l1_tile_idx_1 -> UNPACKER1 -> SRCB
 */
inline void _llk_unpack_binary_operands_(const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1)
{
    // RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users input offset idx

    // Reset Dest counters for Unpacker0/1 to 0
    // Set Source counter to L1 base + offset
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, start_l1_tile_idx_0);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, start_l1_tile_idx_1);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
