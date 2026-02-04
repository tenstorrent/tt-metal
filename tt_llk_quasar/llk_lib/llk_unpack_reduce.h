// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @brief MOP configuration for unpack reduce operations
 * @details Sets up MOP for unpacking for reduce operations, which unpacks
 * tile for SrcA, and a single face for SrcB
 * @tparam REDUCE_DIM: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * buf_desc_id_0 will be used for UNPACKER0 -> SRCA
 * buf_desc_id_1 will be used for UNPACKER1 -> SRCB
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for SrcA, SrcB will only have first face unpacked
 * @param tile_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <ReduceDim REDUCE_DIM>
inline void _llk_unpack_reduce_mop_config_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t num_tiles, const TileShape& tile_shape)
{
    const std::uint32_t MOP_OUTER_LOOP = num_tiles;
    const std::uint32_t MOP_INNER_LOOP = tile_shape.num_faces;

    std::uint32_t unpack_srcA_face = TT_OP_UNPACR0_FACE_INC(0, 1 /*Src face Idx*/, 0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);
    std::uint32_t unpack_srcB_face = TT_OP_UNPACR1_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_srcA_face);

    if constexpr (REDUCE_DIM == ReduceDim::REDUCE_SCALAR)
    {
        // Need to zero out srcA first, because math will do some copying over to SrcA later
        constexpr static std::uint32_t unpack_zero_srcA =
            TT_OP_UNPACR_NOP(p_unpacr::UNP_A, 0, p_unpacr::UNP_STALL_UNP_WR, 0, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC_ZERO);
        temp.set_loop_instr(unpack_zero_srcA, unpack_srcA_face);
    }

    temp.set_start_op(unpack_srcB_face);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief MOP configuration for unpack reduce operations
 * @details Sets up MOP for unpacking for reduce operations, which unpacks
 * tile for SrcA, and a single face for SrcB
 * @tparam REDUCE_DIM: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * buf_desc_id_0 will be used for UNPACKER0 -> SRCA
 * buf_desc_id_1 will be used for UNPACKER1 -> SRCB
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for SrcA, SrcB will only have first face unpacked
 * @param tile_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <ReduceDim REDUCE_DIM>
inline void _llk_unpack_reduce_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t num_tiles, const TileShape& tile_shape)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, (REDUCE_DIM == ReduceDim::REDUCE_ROW));

    _llk_unpack_reduce_mop_config_<REDUCE_DIM>(buf_desc_id_0, buf_desc_id_1, num_tiles, tile_shape);
}

/**
 * @brief Unpacks for reduce kernels
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer
 * start_l1_tile_idx_0 -> UNPACKER0 -> SRCA
 * start_l1_tile_idx_1 -> UNPACKER1 -> SRCB
 */
inline void _llk_unpack_reduce_(const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1)
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
