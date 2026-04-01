// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_pack_common.h"

using namespace ckernel;

/**
 * @brief MOP configuration for pack of contiguous tiles
 * @details Sets up MOP for packing out tile by tile from the math destination register via Packer 0.
 * Packer 1 (SrcS / PACR1) is not supported here: it requires autoloop programming; use
 * _llk_pack_srcs_config_ / _llk_pack_srcs_ in llk_srcs_tdma.h instead.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param num_tiles: number of tiles to pack at a time
 */
inline void _llk_pack_mop_config_(const std::uint8_t buf_desc_id, const std::uint32_t num_tiles)
{
    const std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP = num_tiles;

    // RT: Use defines to remove these constexpr, and replace with a single TT_OP_PACR_FACE_INC
    const std::uint32_t pack_instrn = TT_OP_PACR0_TILE_INC(1 /*Dest Tile Idx*/, 1 /*Src Tile Idx*/, buf_desc_id, 0);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialization for pack of contiguous tiles
 * @details Sets up MOP for packing out tile by tile via Packer 0.
 * Optionally programs packer ReLU (MODE and THRESHOLD) for Packer 0 via cfg_rmw.
 * @tparam EN_32B_DEST: Set to true when pack reads from dst register in Float32;
 * controls RELU_THRESHOLD register format (32-bit or 16-bit path).
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param num_tiles: number of tiles to pack at a time
 * @param relu_config ReLU config (mode + threshold).
 */
template <bool EN_32B_DEST = false>
inline void _llk_pack_init_(
    const std::uint8_t buf_desc_id, const std::uint32_t num_tiles = NUM_TILES, const ckernel::ReluConfig& relu_config = ckernel::ReluConfig::none())
{
    _llk_pack_mop_config_(buf_desc_id, num_tiles);
    _llk_pack_relu_config_<p_pacr::PACK0, EN_32B_DEST>(relu_config);
}

/**
 * @brief Packs out tiles from the math destination register via Packer 0
 * @param start_math_dest_tile_idx: The tile index into the math destination register
 * that packer can start packing from
 * @param start_l1_tile_idx: The tile index into the l1 output buffer
 * that packer can start packing into
 */
inline void _llk_pack_(const std::uint32_t start_math_dest_tile_idx, const std::uint32_t start_l1_tile_idx)
{
    //(TODO) RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users offset idx

    // Set Source (math destination) counter to face index offset
    // Set dst (l1 output) counter to face index offset
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, start_math_dest_tile_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, start_l1_tile_idx);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
