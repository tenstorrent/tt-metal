// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cunpack_common.h"
#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @brief MOP configuration for upk tilize for full 32x32 tiles using the fused HW instruction
 * @details Sets up MOP for unpacking and tilizing a single tile, works for SRCA/B/S and DEST
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_S
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32b mode
 */
template <uint32_t UNP_SEL, uint32_t BUF_DESC_ID, bool IS_32b_DEST_EN, uint32_t BLOCK_CT_DIM>
inline void _llk_unpack_tilize_mop_config_()
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_DEST");
    static_assert((BUF_DESC_ID < 16 && BUF_DESC_ID >= 0), "BUF_DESC_ID should be between 0-16 for unpackers");

    constexpr uint32_t MOP_OUTER_LOOP = 1;
    constexpr uint32_t MOP_INNER_LOOP = BLOCK_CT_DIM;

    constexpr static uint unpack_tile_instrn = TT_OP_UNPACR_TILIZE(0, 0, 0 /*dst Z increment*/, 1 /*src Z increment*/, UNP_SEL, BUF_DESC_ID, 1 /*Set Dvalid*/);

    constexpr static uint reset_src_reg_instrn =
        TT_OP_UNPACR_TILIZE(0, 1 /*Cntr_Reset_Mask*/, 0 /*dst Z increment*/, 0 /*src Z increment*/, UNP_SEL, BUF_DESC_ID, 1 /*Set Dvalid*/);

    if constexpr (IS_32b_DEST_EN)
    {
        // FP32 datacopy uses ELWADD, which requires dvalid from both SrcA and SrcB
        // Set dvalid for the opposite unpacker (if using UNP_A, set dvalid for UNP_B and vice versa)
        constexpr uint32_t OPPOSITE_UNP                  = (UNP_SEL == p_unpacr::UNP_A) ? p_unpacr::UNP_B : p_unpacr::UNP_A;
        constexpr static uint set_opposite_dvalid_instrn = TT_OP_UNPACR_NOP(OPPOSITE_UNP, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*UNP_CLR_SRC*/);

        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, set_opposite_dvalid_instrn, unpack_tile_instrn);
        temp.set_last_outer_loop_instr(reset_src_reg_instrn);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);
        temp.set_last_outer_loop_instr(reset_src_reg_instrn);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Initialized unpacker to unpack tilize a single operand by full 32x32 tiles
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32b mode
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet
 * @tparam C_DIM_FACES: number of faces in c_dim = number of tiles in c_dim * faces in c_dim per tile
 */
template <uint32_t UNP_SEL, uint32_t BUF_DESC_ID, bool IS_32b_DEST_EN, uint32_t FULL_CT_DIM, uint32_t BLOCK_CT_DIM, uint32_t C_DIM_FACES>
inline void _llk_unpack_tilize_init_()
{
    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);                            // Disable transpose
        cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_RMW, C_DIM_FACES); // col dim of a tile in L1 in units of 16 datums (1 face). This is used for
                                                                                   // Src (L1) counter increments in the UNPACR_TILIZE instruction
        cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_TILIZE_DST_Z_STRIDE_RMW, 1); // col dim of a tile in SRC reg - SRC reg will always be 16 datums across (1 face)
        cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
        cfg_rmw(THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_RMW, FULL_CT_DIM * C_DIM_FACES); // how much to stride to go to next row within the same tile
    }
    else
    {
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);                            // Disable transpose
        cfg_rmw(THCON_UNPACKER1_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_RMW, C_DIM_FACES); // col dim of a tile in L1 in units of 16 datums (1 face). This is used for
                                                                                   // Src (L1) counter increments in the UNPACR_TILIZE instruction
        cfg_rmw(THCON_UNPACKER1_REG1_UNPACK_TILIZE_DST_Z_STRIDE_RMW, 1); // col dim of a tile in SRC reg - SRC reg will always be 16 datums across (1 face)
        cfg_rmw(THCON_UNPACKER1_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
        cfg_rmw(THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_RMW, FULL_CT_DIM * C_DIM_FACES); // how much to stride to go to next row within the same tile
    }
    _llk_unpack_tilize_mop_config_<UNP_SEL, BUF_DESC_ID, IS_32b_DEST_EN, BLOCK_CT_DIM>();
}

/**
 * @brief Unpacks a single full 32x32 tile, works for UNP_A, UNP_B, UNP_DEST
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @param l1_tile_idx: Index into the L1 buffer for a tile
 */
template <uint32_t UNP_SEL>
inline void _llk_unpack_tilize_(const uint l1_tile_idx)
{
    // RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users input offset idx

    // Reset Dest counters for Unpacker to 0
    // Set Source counter to L1 base + offset
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, UNP_SEL, l1_tile_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 0);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief MOP configuration for upk tilize strided, works for 32x32 and 16x32 tiles
 * @details Sets up MOP for unpacking half a face with strided instruction and increments L1 counter
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32b mode
 * @param c_dim_face: number of faces in c_dim = number of tiles in c_dim * faces in c_dim per tile
 */
template <uint32_t UNP_SEL, uint32_t BUF_DESC_ID, bool IS_32b_DEST_EN, uint32_t FULL_CT_DIM, uint32_t C_DIM_FACES>
inline void _llk_unpack_tilize_strided_mop_config_()
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_DEST");
    static_assert((BUF_DESC_ID < 16 && BUF_DESC_ID >= 0), "BUF_DESC_ID should be between 0-16 for unpackers");

    constexpr uint32_t MOP_OUTER_LOOP = 1;
    constexpr uint32_t MOP_INNER_LOOP = 1;

    constexpr static uint unpack_half_face_instrn =
        (UNP_SEL == p_unpacr::UNP_A)
            ? TT_OP_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, BUF_DESC_ID, 0 /*Set Dvalid*/)
            : TT_OP_UNPACR1_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, BUF_DESC_ID, 0 /*Set Dvalid*/);
    constexpr static uint increment_half_face_instrn =
        TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, C_DIM_FACES * ckernel::unpack::UNPACR_STRIDE_MAX_ROWS * FULL_CT_DIM);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_half_face_instrn, increment_half_face_instrn);
    temp.set_end_op(unpack_half_face_instrn);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialized unpacker with stride values for tilize strided 32x32 or 16x32
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32b mode
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet
 * @tparam C_DIM_FACES: number of faces in c_dim = number of tiles in c_dim * faces in c_dim per tile
 */
template <uint32_t UNP_SEL, uint32_t BUF_DESC_ID, bool IS_32b_DEST_EN, uint32_t FULL_CT_DIM, uint32_t C_DIM_FACES>
inline void _llk_unpack_tilize_strided_init_()
{
    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0); // Disable transpose
        cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
        cfg_rmw(THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_RMW, FULL_CT_DIM * C_DIM_FACES);
    }
    else
    {
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0); // Disable transpose
        cfg_rmw(THCON_UNPACKER1_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
        cfg_rmw(THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_RMW, FULL_CT_DIM * C_DIM_FACES);
    }
    _llk_unpack_tilize_strided_mop_config_<UNP_SEL, BUF_DESC_ID, IS_32b_DEST_EN, FULL_CT_DIM, C_DIM_FACES>(); // TODO: This throws a compile time error for UPK
                                                                                                              // but not PCK - why?
}

/**
 * @brief Unpacks an entire tile, using the stride instruction to tilize for 32x32 or 16x32 tiles
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32b mode
 * @tparam C_DIM_FACES: number of faces in c_dim = number of tiles in c_dim * faces in c_dim per tile
 * @param tile_shape: xyz dimensions of the tile
 * @param l1_tile_idx: c_dim index of tile in L1
 */
template <uint32_t UNP_SEL, uint32_t FULL_CT_DIM, bool IS_32b_DEST_EN, uint32_t C_DIM_FACES>
inline void _llk_unpack_tilize_strided_(const TileShape& tile_shape, const uint l1_tile_idx)
{
    const uint32_t f1_row_idx =
        (tile_shape.narrow_tile ? l1_tile_idx * C_DIM_FACES + C_DIM_FACES * tile_shape.face_r_dim * FULL_CT_DIM : l1_tile_idx * C_DIM_FACES + 1);

    // Reset Dest counters for Unpacker to 0
    // Set Source counter to L1 base + offset
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, l1_tile_idx * C_DIM_FACES);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 0);

    // Face 0
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, f1_row_idx); // Set L1 ptr

    if (tile_shape.num_faces == ckernel::trisc::NUM_FACES)
    {
        // Face 1
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer); // Unpack half face and increment L1 ptr

        // Face 2
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, l1_tile_idx * C_DIM_FACES + C_DIM_FACES * tile_shape.face_r_dim * FULL_CT_DIM);
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

        // Face 3
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, l1_tile_idx * C_DIM_FACES + C_DIM_FACES * tile_shape.face_r_dim * FULL_CT_DIM + 1);
    }
    if constexpr (UNP_SEL == p_unpacr::UNP_A && IS_32b_DEST_EN)
    { // TODO pgardner: I dont think ill need this for float32 dest
        TTI_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/);
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B && IS_32b_DEST_EN)
    {
        TTI_UNPACR_NOP(p_unpacr::UNP_A, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/);
    }

    if constexpr (UNP_SEL == p_unpacr::UNP_A)
    { // This is ugly but will likely change soon anyways
        TTI_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, 0, 0 /*Set Dvalid*/);
        TTI_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, C_DIM_FACES * ckernel::unpack::UNPACR_STRIDE_MAX_ROWS * FULL_CT_DIM);
        TTI_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, 0, 1 /*Set Dvalid*/);
    }
    else
    {
        TTI_UNPACR1_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, 0, 0 /*Set Dvalid*/);
        TTI_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, C_DIM_FACES * ckernel::unpack::UNPACR_STRIDE_MAX_ROWS * FULL_CT_DIM);
        TTI_UNPACR1_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, 0, 1 /*Set Dvalid*/);
    }
}

/**
 * @brief MOP configuration for upk tilize strided for tiles size 8x32, 4x32, 2x32, 1x32
 * @details Sets up MOP for unpacking half a face with strided instruction and increments L1 counter
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32b mode
 * @tparam ROWS_READ: number of rows read by one UNPACR0_STRIDE call
 */
template <uint32_t UNP_SEL, uint32_t BUF_DESC_ID, bool IS_32b_DEST_EN, uint32_t FULL_CT_DIM, uint32_t ROWS_READ>
inline void _llk_unpack_tilize_strided_mop_config_small_faces_()
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_DEST");
    static_assert((BUF_DESC_ID < 16 && BUF_DESC_ID >= 0), "BUF_DESC_ID should be between 0-16 for unpackers");
    static_assert(!(IS_32b_DEST_EN && UNP_SEL != p_unpacr::UNP_A), "If IS_32b_DEST_EN then UNP_SEL should be UNP_A");

    constexpr uint32_t MOP_OUTER_LOOP = 1;
    constexpr uint32_t MOP_INNER_LOOP = FULL_CT_DIM;

    constexpr static uint unpack_face0_instrn =
        TT_OP_UNPACR0_STRIDE(ROWS_READ /*Src_Reg_Y_Cntr_Incr*/, 0 /*inc by 1*/, 1 /*set to inc*/, 0, 0, BUF_DESC_ID, 0 /*Set Dvalid*/);
    constexpr static uint unpack_face1_instrn = TT_OP_UNPACR0_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0, 1, 0, 0, BUF_DESC_ID, 1 /*Set Dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_face0_instrn, unpack_face1_instrn);

    // FP32 datacopy uses ELWADD, which requires datavalid from both SrcA and SrcB, so need to add SrcB datavalid
    if constexpr (UNP_SEL == p_unpacr::UNP_A && IS_32b_DEST_EN)
    {
        temp.set_end_op(TT_OP_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/));
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B && IS_32b_DEST_EN)
    {
        temp.set_end_op(TT_OP_UNPACR_NOP(p_unpacr::UNP_A, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/));
    }

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialized unpacker with stride values for tilize strided for 8x32, 4x32, 2x32, 1x32
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32b mode
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet
 * @tparam ROWS_READ: number of rows read by one UNPACR0_STRIDE call
 * @tparam C_DIM_FACES: number of faces in c_dim = number of tiles in c_dim * faces in c_dim per tile
 */
template <uint32_t UNP_SEL, uint32_t BUF_DESC_ID, bool IS_32b_DEST_EN, uint32_t FULL_CT_DIM, uint32_t ROWS_READ, uint32_t C_DIM_FACES>
inline void _llk_unpack_tilize_strided_init_small_faces_()
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0); // Disable transpose
    cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);
    cfg_rmw(THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_RMW, FULL_CT_DIM * C_DIM_FACES);
    _llk_unpack_tilize_strided_mop_config_small_faces_<UNP_SEL, BUF_DESC_ID, IS_32b_DEST_EN, FULL_CT_DIM, ROWS_READ>();
}

/**
 * @brief Unpacks an entire tiny tile, using the stride instruction to tilize. Does sizes 8x32, 4x32, 2x32, 1x32
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_S
 * @tparam FULL_CT_DIM: Number of tiles in a row of the input tensor. Input tensor is row-major format. R_DIM not implemented yet
 * @param tile_shape: xyz dimensions of the tile
 * @param l1_tile_idx: c_dim index of tile in L1
 * @param c_dim_face: number of faces in c_dim = number of tiles in c_dim * faces in c_dim per tile
 */
template <uint32_t UNP_SEL, uint32_t FULL_CT_DIM>
inline void _llk_unpack_tilize_strided_small_faces_(const TileShape& tile_shape)
{
    // Reset Dest counters for Unpacker to 0
    // Set Source counter to L1 base + offset

    // Face 0 & 1
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
