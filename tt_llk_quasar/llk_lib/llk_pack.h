// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "llk_pack_common.h"

using namespace ckernel;

/**
 * @brief MOP configuration for pack of contiguous tiles
 * @details Sets up MOP for packing out tile by tile works for any pack resource
 * @tparam PACK_SEL: Selects which unpacker resource to use,
 * values = p_pacr::PACK0/PACK1
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param num_tiles: number of tiles to pack at a time
 */
template <uint8_t PACK_SEL>
inline void _llk_pack_mop_config_(const uint8_t buf_desc_id, const uint32_t num_tiles)
{
    static_assert((PACK_SEL == p_pacr::PACK0) || (PACK_SEL == p_pacr::PACK1), "PACK_SEL can only be set to p_pacr::PACK0/PACK1");

    const uint32_t MOP_OUTER_LOOP = 1;
    const uint32_t MOP_INNER_LOOP = num_tiles;

    // RT: Use defines to remove these constexpr, and replace with a single TT_OP_PACR_FACE_INC
    uint pack_instrn;
    if constexpr (PACK_SEL == p_pacr::PACK0)
    {
        pack_instrn = TT_OP_PACR0_TILE_INC(1 /*Dest Tile Idx*/, 1 /*Src Tile Idx*/, buf_desc_id, 0);
    }
    else
    {
        // PACR1 slice size is only 16x8, no need to increment SrcS counter
        pack_instrn = TT_OP_PACR1_TILE_INC(1 /*Dest Tile Idx*/, 0, buf_desc_id, 0);
    }

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialization for pack of contiguous tiles
 * @details Sets up MOP for packing out tile by tile works for any pack resource
 * @tparam PACK_SEL: Selects which unpacker resource to use,
 * values = p_pacr::PACK0/PACK1
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param num_tiles: number of tiles to pack at a time
 */
template <uint8_t PACK_SEL>
inline void _llk_pack_init_(const uint8_t buf_desc_id, const uint32_t num_tiles)
{
    _llk_pack_mop_config_<PACK_SEL>(buf_desc_id, num_tiles);
}

/**
 * @brief Packs out tiles, works for either pack resource
 * @tparam PACK_SEL: Selects which packer resource to use,
 * values = p_pacr::PACK0/PACK1
 * @param start_math_dest_tile_idx: The tile index into the math destination register
 * that packer can start packing from
 * @param start_l1_tile_idx: The tile index into the l1 output buffer
 * that packer can start packing into
 */
template <uint8_t PACK_SEL>
inline void _llk_pack_(
    const uint start_math_dest_tile_idx, const uint start_l1_tile_idx

)
{
    //(TODO) RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users offset idx

    // Set Source (math destination) counter to face index offset
    // Set dst (l1 output) counter to face index offset
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, PACK_SEL, start_math_dest_tile_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, PACK_SEL, start_l1_tile_idx);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
