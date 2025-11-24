// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @brief MOP configuration for unpack of binary operations with broadcasts, uses SrcA & SrcB
 * @details Sets up MOP for unpacking binary operands tile by tile
 * BUF_DESC_ID_0 will be used for UNPACKER0 -> SRCA
 * BUF_DESC_ID_1 will be used for UNPACKER1 -> SRCB
 * @tparam BUF_DESC_ID_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam BROADCAST_TYPE: Sets the broadcast type, values = [NONE, COL, ROW, SCALAR]
 * BROADCAST only operates on SRCB register
 * @param num_tiles: number of tiles to unpack at a time for both inputs
 */
template <uint32_t BUF_DESC_ID_0, uint32_t BUF_DESC_ID_1, BroadcastType BROADCAST_TYPE>
inline void _llk_unpack_binary_broadcast_operands_mop_config_(const uint32_t num_tiles)
{
    static_assert((BUF_DESC_ID_0 < 16 && BUF_DESC_ID_0 >= 0), "BUF_DESC_ID_0 should be between 0-16 for unpackers");
    static_assert((BUF_DESC_ID_1 < 16 && BUF_DESC_ID_1 >= 0), "BUF_DESC_ID_1 should be between 0-16 for unpackers");
    static_assert((BROADCAST_TYPE != BroadcastType::NONE), "Broadcast type cannot be NONE for this operation");

    const uint32_t MOP_OUTER_LOOP     = num_tiles;
    constexpr uint32_t MOP_INNER_LOOP = 1;

    constexpr static uint unpack_srca_tile_inc = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src Tile Idx*/, BUF_DESC_ID_0, 1 /*Set Dvalid*/);
    constexpr static uint replay_buf_len       = (BROADCAST_TYPE == BroadcastType::SCALAR) ? 1 : 4;

    load_replay_buf<0, replay_buf_len>(
        []
        {
            // Unpacks face 0 into dest offset 0
            TTI_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, BUF_DESC_ID_1, 1 /*Set Dvalid*/);

            if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
            {
                TTI_UNPACR1_FACE(0 /*Dst Face Idx*/, 1 /*Src Face Idx*/, 0, 0, BUF_DESC_ID_1, 1 /*Set Dvalid*/);
                TTI_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, BUF_DESC_ID_1, 1 /*Set Dvalid*/);
                TTI_UNPACR1_FACE(0 /*Dst Face Idx*/, 1 /*Src Face Idx*/, 0, 0, BUF_DESC_ID_1, 1 /*Set Dvalid*/);
            }
            else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
            {
                TTI_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, BUF_DESC_ID_1, 1 /*Set Dvalid*/);
                TTI_UNPACR1_FACE(0 /*Dst Face Idx*/, 2 /*Src Face Idx*/, 0, 0, BUF_DESC_ID_1, 1 /*Set Dvalid*/);
                TTI_UNPACR1_FACE(0 /*Dst Face Idx*/, 2 /*Src Face Idx*/, 0, 0, BUF_DESC_ID_1, 1 /*Set Dvalid*/);
            }
        });

    ckernel_template temp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0),
        TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 1)); // Inc SrcB by 1 tile, because above UNPACR1_FACE does not inc counters

    temp.set_start_op(unpack_srca_tile_inc);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialization for unpack of binary broadcast operations, uses SrcA & SrcB
 * @details Sets up MOP for unpacking binary operands
 * BUF_DESC_ID_0 will be used for UNPACKER0 -> SRCA
 * BUF_DESC_ID_1 will be used for UNPACKER1 -> SRCB
 * @tparam BUF_DESC_ID_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam BROADCAST_TYPE: Sets the broadcast type, values = [NONE, COL, ROW, SCALAR]
 * BROADCAST only operates on SRCB register
 * @param num_tiles: number of tiles to unpack at a time for both inputs
 */
template <uint32_t BUF_DESC_ID_0, uint32_t BUF_DESC_ID_1, BroadcastType BROADCAST_TYPE>
inline void _llk_unpack_binary_broadcast_operands_init_(const uint32_t num_tiles)
{
    _llk_unpack_binary_broadcast_operands_mop_config_<BUF_DESC_ID_0, BUF_DESC_ID_1, BROADCAST_TYPE>(num_tiles);
}

/**
 * @brief Unpacks binary operands for SrcA & SrcB
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer
 * start_l1_tile_idx_0 -> UNPACKER0 -> SRCA
 * start_l1_tile_idx_1 -> UNPACKER1 -> SRCB
 */
inline void _llk_unpack_binary_broadcast_operands_(const uint start_l1_tile_idx_0, const uint start_l1_tile_idx_1)
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
