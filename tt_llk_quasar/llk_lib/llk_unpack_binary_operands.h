// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @brief MOP configuration for unpack of binary operations, uses SrcA & SrcB
 * @details Sets up MOP for unpacking binary operands tile by tile
 * BUF_DESC_ID_0 will be used for UNPACKER0 -> SRCA
 * BUF_DESC_ID_1 will be used for UNPACKER1 -> SRCB
 * @tparam BUF_DESC_ID_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for both inputs
 */
template <uint32_t BUF_DESC_ID_0, uint32_t BUF_DESC_ID_1>
inline void _llk_unpack_binary_operands_mop_config_(const uint32_t num_tiles)
{
    static_assert((BUF_DESC_ID_0 < 16 && BUF_DESC_ID_0 >= 0), "BUF_DESC_ID_0 should be between 0-16 for unpackers");
    static_assert((BUF_DESC_ID_1 < 16 && BUF_DESC_ID_1 >= 0), "BUF_DESC_ID_1 should be between 0-16 for unpackers");

    constexpr uint32_t MOP_OUTER_LOOP = 1;
    const uint32_t MOP_INNER_LOOP     = num_tiles;

    constexpr static uint unpack_instrn0 = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src Tile Idx*/, BUF_DESC_ID_0, 1 /*Set Dvalid*/);
    constexpr static uint unpack_instrn1 = TT_OP_UNPACR1_TILE_INC(0, 1 /*Src Tile Idx*/, BUF_DESC_ID_1, 1 /*Set Dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_instrn0, unpack_instrn1);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialization for unpack of binary operations, uses SrcA & SrcB
 * @details Sets up MOP for unpacking binary operands
 * BUF_DESC_ID_0 will be used for UNPACKER0 -> SRCA
 * BUF_DESC_ID_1 will be used for UNPACKER1 -> SRCB
 * @tparam BUF_DESC_ID_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for both inputs
 */
template <uint32_t BUF_DESC_ID_0, uint32_t BUF_DESC_ID_1>
inline void _llk_unpack_binary_operands_init_(const uint32_t num_tiles)
{
    _llk_unpack_binary_operands_mop_config_<BUF_DESC_ID_0, BUF_DESC_ID_1>(num_tiles);
}

/**
 * @brief Unpacks binary operands for SrcA & SrcB
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer
 * start_l1_tile_idx_0 -> UNPACKER0 -> SRCA
 * start_l1_tile_idx_1 -> UNPACKER1 -> SRCB
 */
inline void _llk_unpack_binary_operands_(const uint start_l1_tile_idx_0, const uint start_l1_tile_idx_1)
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
