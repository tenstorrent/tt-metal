// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_pack_common.h"

using namespace ckernel;

/**
 * @brief MOP configuration for matmul pack of contiguous tiles
 * @details Sets up MOP for packing out tile by tile via Packer 0. Packer 1 (SrcS) uses llk_srcs_tdma.h.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param subblock_r_dim: number of tiles in the row dimension of a matrix block
 * @param subblock_c_dim: number of tiles in the column dimension of a matrix block
 * @param num_subblocks_c_dim: number of subblocks in the column dimension of a matrix block
 */
inline void _llk_pack_matmul_mop_config_(
    const std::uint32_t buf_desc_id, const std::uint32_t subblock_r_dim, const std::uint32_t subblock_c_dim, const std::uint32_t num_subblocks_c_dim)
{
    const std::uint32_t MOP_OUTER_LOOP = subblock_r_dim;
    const std::uint32_t MOP_INNER_LOOP = subblock_c_dim;

    // RT: Use defines to remove these constexpr, and replace with a single TT_OP_PACR_FACE_INC
    std::uint32_t pack_instrn = TT_OP_PACR0_TILE_INC(1 /*Dst (l1) tile idx*/, 1 /*Src tile Idx*/, buf_desc_id, 0);
    std::uint32_t incr_l1_ptr = TT_OP_INC_DST_TILE_FACE_ROW_IDX(
        p_set_inc_sel::TILE_SEL, p_pacr::PACK0, subblock_c_dim * num_subblocks_c_dim - subblock_c_dim); // cycle pipelined by PACR0_TILE_INC taking >=8 cycles
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn);
    temp.set_end_op(incr_l1_ptr);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialization for matmul pack of contiguous tiles (Packer 0)
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param subblock_r_dim: number of tiles in the row dimension of a matrix block
 * @param subblock_c_dim: number of tiles in the column dimension of a matrix block
 * @param num_subblocks_c_dim: number of subblocks in the column dimension of a matrix block
 */
inline void _llk_pack_matmul_init_(
    const std::uint32_t buf_desc_id, const std::uint32_t subblock_r_dim, const std::uint32_t subblock_c_dim, const std::uint32_t num_subblocks_c_dim)
{
    _llk_pack_matmul_mop_config_(buf_desc_id, subblock_r_dim, subblock_c_dim, num_subblocks_c_dim);
}

/**
 * @brief Packs out tiles via Packer 0 (matmul MOP)
 * @param start_math_dest_tile_idx: The tile index into the math destination register
 * that packer can start packing from
 * @param start_l1_tile_idx: The tile index into the l1 output buffer
 * that packer can start packing into
 */
inline void _llk_pack_matmul_(const std::uint32_t start_math_dest_tile_idx, const std::uint32_t start_l1_tile_idx)
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
