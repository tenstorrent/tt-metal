// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cpack_common.h"
#include "llk_pack_common.h"

using namespace ckernel;

/**
 * @brief MOP configuration for pack untilize of contiguous tiles
 * @details Sets up MOP for packing out tile by tile and untilizing, works only with PACR0
 * @tparam BLOCK_CT_DIM: c_dim of tiles in each block (<= 8)
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 */
template <std::uint32_t BLOCK_CT_DIM>
inline void _llk_pack_untilize_mop_config_(const std::uint8_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = BLOCK_CT_DIM;

    std::uint32_t pack_instrn = TT_OP_PACR_UNTILIZE(0, 0, 1 /*inc Dst (L1) ctr*/, 1 /*inc Src ctr*/, 0 /*Packer 0 Sel*/, buf_desc_id, 0);
    // reset_src_and_dst_reg_instrn will reset Src (DEST reg) and Dst (L1) counters when the DEST reg bank is flipped aka at the last tile of the
    // block
    std::uint32_t reset_src_and_dst_reg_instrn =
        TT_OP_PACR_UNTILIZE(0, 3 /*Cntr_Reset_Mask*/, 0 /*inc Dst (L1) ctr*/, 0 /*inc Src ctr*/, 0, buf_desc_id, 0 /*Clr Dvalid*/);
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn);
    temp.set_last_outer_loop_instr(reset_src_and_dst_reg_instrn);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialization for pack untilize of contiguous tiles
 * @details Sets up cfg registers with tile & tensor shape parameters which inform how TT_OP_PACR_UNTILIZE will read the data
 * @tparam FULL_CT_DIM: number of tiles per block in c_dim * number of blocks = full c_dim size of tensor
 * @tparam BLOCK_CT_DIM: c_dim of tiles in each block (<= 8)
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param tile_shape: defines shape of tile - used to parameterize for tiny tiles
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t BLOCK_CT_DIM, std::uint32_t C_DIM_FACES>
inline void _llk_pack_untilize_init_(const std::uint8_t buf_desc_id, const TileShape& tile_shape)
{
    cfg_rmw(THCON_PACKER0_REG1_PACK_UNTILIZE_SRC_Z_STRIDE_RMW, tile_shape.num_faces * tile_shape.face_r_dim); // inc MATH DEST REG ptr by 64 16-datum rows
    cfg_rmw(THCON_PACKER0_REG1_PACK_UNTILIZE_DST_Z_STRIDE_RMW, C_DIM_FACES);                                  // inc L1 SRC ptr by 2 16-datum rows
    cfg_rmw(THCON_PACKER0_REG1_PACK_STRIDE_OFFSET_0_RMW, C_DIM_FACES * FULL_CT_DIM);                          // stride each row by 2*C_DIM 16-datum rows
    _llk_pack_untilize_mop_config_<BLOCK_CT_DIM>(buf_desc_id);
}

/**
 * @brief Packs out tiles and untilizes, always use PCK0 for untilize
 * @param dest_idx: Index into the DEST register for a tile
 * @param l1_tile_idx: Index into the L1 buffer for a tile
 */
inline void _llk_pack_untilize_(const std::uint32_t dest_idx, const std::uint32_t l1_tile_idx)
{
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, dest_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_pacr::PACK0, l1_tile_idx);
    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief MOP configuration for partial pack of a face
 * @details Sets up MOP for packing out part of a face using strided reads to untilize
 * @tparam FULL_CT_DIM: number of tiles per block in c_dim * number of blocks = full c_dim size of tensor
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t FACE_R_DIM, std::uint32_t C_DIM_FACES>
inline void _llk_pack_untilize_strided_mop_config_(const std::uint8_t buf_desc_id)
{
    constexpr std::uint32_t ROWS_READ      = 4;
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = FACE_R_DIM == 16 ? 3 : 1;

    std::uint32_t pack_instrn           = TT_OP_PACR_STRIDE(1, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
    constexpr std::uint32_t incr_L1_ptr = TT_OP_INC_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, ROWS_READ * C_DIM_FACES * FULL_CT_DIM);
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn, incr_L1_ptr);
    // temp.set_end_op(pack_instrn);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief MOP configuration for partial pack of a face
 * @details Sets up MOP for packing out part of a face using strided reads to untilize
 * @tparam FULL_CT_DIM: number of tiles per block in c_dim * number of blocks = full c_dim size of tensor
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t NUM_TILES_PER_BLOCK, std::uint32_t C_DIM_FACES>
inline void _llk_pack_untilize_strided_4x32_mop_config_(const std::uint8_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = NUM_TILES_PER_BLOCK;

    constexpr static std::uint32_t reset_dest_reg_ptr = TT_OP_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 0);
    std::uint32_t pack_instrn                         = TT_OP_PACR_STRIDE(1, 1, 0, 1, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
    std::uint32_t pack_instrn_rest_reg_ctr            = TT_OP_PACR_STRIDE(0, 1, 0, 1, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn, pack_instrn);
    temp.set_start_op(reset_dest_reg_ptr);
    temp.set_last_outer_loop_instr(pack_instrn_rest_reg_ctr);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

template <std::uint32_t FULL_CT_DIM, std::uint32_t NUM_TILES_PER_BLOCK, std::uint32_t C_DIM_FACES>
inline void _llk_pack_untilize_strided_2x32_mop_config_(const std::uint8_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = NUM_TILES_PER_BLOCK * 2;
    constexpr std::uint32_t replay_buf_len = 2;

    load_replay_buf<0, replay_buf_len>(
        [buf_desc_id]
        {
            // Unpacks face 0 into dest offset 0
            TT_PACR_STRIDE(0, 0, 0, 1, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
            TTI_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 2);
        });

    constexpr static std::uint32_t reset_dest_reg_ptr = TT_OP_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 0);
    std::uint32_t pack_instrn_rest_reg_ctr            = TT_OP_PACR_STRIDE(0, 1, 0, 1, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
    temp.set_start_op(reset_dest_reg_ptr);
    temp.set_last_outer_loop_instr(pack_instrn_rest_reg_ctr);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

template <std::uint32_t FULL_CT_DIM, std::uint32_t NUM_TILES_PER_BLOCK, std::uint32_t C_DIM_FACES>
inline void _llk_pack_untilize_strided_1x32_mop_config_(const std::uint8_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = NUM_TILES_PER_BLOCK;

    std::uint32_t pack_instrn                         = TT_OP_PACR0_ROW(0, 0, 0, 0, 1 /*inc L1 by 1 row*/, 1 /*inc DEST by 1 row*/, buf_desc_id, 0);
    constexpr static std::uint32_t reset_dest_reg_ptr = TT_OP_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 0);
    std::uint32_t pack_instrn_rest_reg_ctr            = TT_OP_PACR0_ROW(0, 0, 0, 0, 1 /*inc L1 by 1 row*/, 0 /*inc DEST by 1 row*/, buf_desc_id, 0);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn, pack_instrn);
    temp.set_start_op(reset_dest_reg_ptr);
    temp.set_last_outer_loop_instr(pack_instrn_rest_reg_ctr);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialization for pack untilize of contiguous tiles using strided pack
 * @details Sets up cfg registers with tile & tensor shape parameters which inform how TT_OP_PACR_STRIDE will read the data
 * @tparam FULL_CT_DIM: number of tiles per block in c_dim * number of blocks = full c_dim size of tensor
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param tile_shape: defines shape of tile - used to parameterize for tiny tiles
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t FACE_R_DIM, std::uint32_t NUM_TILES_PER_BLOCK, std::uint32_t C_DIM_FACES>
inline void _llk_pack_untilize_strided_init_(const std::uint8_t buf_desc_id, const TileShape& tile_shape)
{
    cfg_rmw(THCON_PACKER0_REG3_PACK_STRIDE_VAL_SOURCE_RMW, 0); // sel STRIDE_OFFSET_0
    if constexpr (FACE_R_DIM >= 8)
    {                                                                                    // 32x32 16x32 and 8x32
        cfg_rmw(THCON_PACKER0_REG1_PACK_STRIDE_OFFSET_0_RMW, C_DIM_FACES * FULL_CT_DIM); // stride each row by 2*FULL_CT_DIM 16-datum rows
        _llk_pack_untilize_strided_mop_config_<FULL_CT_DIM, FACE_R_DIM, C_DIM_FACES>(buf_desc_id);
    }
    else if constexpr (FACE_R_DIM == 4)
    {                                                                                    // 4x32
        cfg_rmw(THCON_PACKER0_REG1_PACK_STRIDE_OFFSET_0_RMW, C_DIM_FACES * FULL_CT_DIM); // stride each row by 2*FULL_CT_DIM 16-datum rows
        _llk_pack_untilize_strided_4x32_mop_config_<FULL_CT_DIM, NUM_TILES_PER_BLOCK, C_DIM_FACES>(buf_desc_id);
    }
    else if constexpr (FACE_R_DIM == 2)
    {                                                            // 2x32
        cfg_rmw(THCON_PACKER0_REG3_PACK_STRIDE_NO_WRITE_RMW, 1); // Need to mask off last 2 rows
        cfg_rmw(THCON_PACKER0_REG3_PACK_STRIDE_ROW_MASK_RMW, 0x1100);
        cfg_rmw(THCON_PACKER0_REG1_PACK_STRIDE_OFFSET_0_RMW, C_DIM_FACES * FULL_CT_DIM);
        _llk_pack_untilize_strided_2x32_mop_config_<FULL_CT_DIM, NUM_TILES_PER_BLOCK, C_DIM_FACES>(buf_desc_id);
    }
    else
    { // 1x32
        _llk_pack_untilize_strided_1x32_mop_config_<FULL_CT_DIM, NUM_TILES_PER_BLOCK, C_DIM_FACES>(buf_desc_id);
    }
}

/**
 * @brief Packs a single tile, untilizing it using strided pack
 * @tparam FULL_CT_DIM: number of tiles per block in c_dim * number of blocks = full c_dim size of tensor
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param tile_shape: defines shape of tile - used to parameterize for tiny tiles
 */
template <std::uint32_t FULL_CT_DIM, std::uint32_t C_DIM_FACES>
inline void _llk_pack_untilize_strided_(
    const std::uint8_t buf_desc_id, const TileShape& tile_shape, const std::uint32_t l1_tile_idx, const std::uint32_t src_tile_idx)
{
    const std::uint32_t f1_row_idx =
        tile_shape.narrow_tile ? l1_tile_idx * C_DIM_FACES + C_DIM_FACES * tile_shape.face_r_dim * FULL_CT_DIM : l1_tile_idx * C_DIM_FACES + 1;

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, src_tile_idx * tile_shape.face_r_dim * tile_shape.num_faces);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, l1_tile_idx * C_DIM_FACES);

    // Face 0
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    TT_PACR_STRIDE(1, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);

    // Face 1
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, f1_row_idx);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    if (tile_shape.num_faces == ckernel::trisc::NUM_FACES)
    {
        TT_PACR_STRIDE(1, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
        // Face 2
        TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, l1_tile_idx * C_DIM_FACES + C_DIM_FACES * tile_shape.face_r_dim * FULL_CT_DIM);
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        TT_PACR_STRIDE(1, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);

        // Face 3
        TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, l1_tile_idx * C_DIM_FACES + C_DIM_FACES * tile_shape.face_r_dim * FULL_CT_DIM + 1);
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        TT_PACR_STRIDE(0, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
    }
    else
    {
        TT_PACR_STRIDE(0, 1, 0, 0, 0, buf_desc_id, 0 /*pck0 sel*/, 0 /*dvalid*/);
    }
}

inline void _llk_pack_untilize_strided_4x32_()
{
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
