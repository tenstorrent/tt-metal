// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "llk_pack_common.h"
#include "llk_unpack_common.h"

using namespace ckernel;

/**
 * @brief Configure the auto loop registers for unpacking srcs
 * @tparam INSTRN_COUNT: The number of instructions to place in the auto-loop, auto-loop will then
 * be looped by INSTRN_LOOP_COUNT set in the hw_config
 * @tparam INSTRN_LOOP_COUNT: The number of times to loop the instructions in the auto-loop
 */
template <uint8_t INSTRN_COUNT, uint8_t INSTRN_LOOP_COUNT>
inline void _llk_unpack_srcs_config_()
{
    cfg_rmw(THCON_UNPACKER2_REG0_INSTRN_LOOP_COUNT_RMW, INSTRN_LOOP_COUNT - 1);
    cfg_rmw(THCON_UNPACKER2_REG0_INSTRN_COUNT_RMW, INSTRN_COUNT - 1);
}

template <uint8_t INSTRN_COUNT, uint8_t INSTRN_LOOP_COUNT>
inline void _llk_pack_srcs_config_()
{
    cfg_rmw(THCON_PACKER1_REG0_INSTRN_LOOP_COUNT_RMW, INSTRN_LOOP_COUNT - 1);
    cfg_rmw(THCON_PACKER1_REG0_INSTRN_COUNT_RMW, INSTRN_COUNT - 1);
}

/**
 * @brief Unpacks tiles to SrcS, Each SrcS slice is only 8*16*16bit
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0-31
 * @tparam INSTRN_COUNT: The number of instructions to place in the auto-loop, auto-loop will then
 * be looped by INSTRN_LOOP_COUNT set in the hw_config
 * @param start_l1_tile_idx: The tile index into the l1 input buffer
 * that unpacker can start unpacking from
 */
template <uint8_t BUF_DESC_ID, uint8_t INSTRN_COUNT>
inline void _llk_unpack_srcs_(const uint start_l1_tile_idx

)
{
    // Set src (l1 input) counter to face index offset
    // Tile can only be a maximum of 8x16 rows for 16 bit, 4*16 for 32 bit
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, start_l1_tile_idx);

    for (uint32_t i = 0; i < INSTRN_COUNT; i++)
    {
        TTI_UNPACR2_TILE_INC(0b0, 0b1 /*SrcS l1 increment*/, BUF_DESC_ID, 0b1 /*Set dvalid*/);
    }
}

/**
 * @brief Packs tiles from SrcS to L1, Each SrcS slice is only 8*16*16bit
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0-31
 * @tparam INSTRN_COUNT: The number of instructions to place in the auto-loop, auto-loop will then
 * be looped by INSTRN_LOOP_COUNT set in the hw_config
 * @param start_l1_tile_idx: The tile index into the l1 output buffer
 * that packer can start packing to
 */
template <uint8_t BUF_DESC_ID, uint8_t INSTRN_COUNT>
inline void _llk_pack_srcs_(const uint start_l1_tile_idx

)
{
    // Set dst (l1 output) counter to face index offset
    // Tile can only be a maximum of 8x16 rows for 16 bit, 4*16 for 32 bit
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK1, start_l1_tile_idx);

    for (uint32_t i = 0; i < INSTRN_COUNT; i++)
    {
        TTI_PACR1_TILE_INC(0b1 /*DstS l1 increment*/, 0b0 /*SrcS l1 increment*/, BUF_DESC_ID, 0b1 /*Set dvalid*/);
    }
}
