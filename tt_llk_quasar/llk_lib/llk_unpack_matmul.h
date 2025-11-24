// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @brief Initializes unpacker to unpack operand 0 (BUF_DESC_ID_0) into SrcB
 * and unpacks operand 1 (BUF_DESC_ID_1) into SrcA. Matrix multiply FPU operation does SrcB * SrcA.
 * In order to get output of rowmajor matrix multiplication input 0 * Input 1, need to initialize
 * SrcA and SrcB to be input 1 & input 0 respectively.
 * The following matrix multiply has the following dimensions:
 * IMPORTANT NOTE:
 * This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]
 * kt_dim is assumed to be iterated over outside this api call
 * ct_dim * rt_dim <= 8 tiles in Float16b, ct_dim * rt_dim <= 4 tiles in Float32
 * @tparam BUF_DESC_ID_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 32
 * @tparam CT_DIM: number of tiles in the column dimension for input1 of matrix multiply
 * @tparam RT_DIM: number of tiles in the row dimension for input0 of matrix multiply
 * @tparam KT_DIM: number of tiles in the common dimension between input0 & input1 of matrix multiply
 */
template <uint32_t BUF_DESC_ID_0, uint32_t BUF_DESC_ID_1, uint8_t CT_DIM, uint8_t RT_DIM, uint32_t KT_DIM>
inline void _llk_unpack_matmul_mop_config_()
{
    static_assert((BUF_DESC_ID_0 < 32 && BUF_DESC_ID_0 >= 0), "BUF_DESC_ID_0 should be between 0-32 for unpackers");
    static_assert((BUF_DESC_ID_1 < 32 && BUF_DESC_ID_1 >= 0), "BUF_DESC_ID_1 should be between 0-32 for unpackers");

    constexpr bool reuse_a            = CT_DIM >= RT_DIM;
    constexpr uint32_t MOP_OUTER_LOOP = 1;
    constexpr uint32_t MOP_INNER_LOOP = reuse_a ? CT_DIM : RT_DIM;
    static uint unpack_instrn;
    // static uint inc_l1_instrn;
    static uint unpack_reuse_instrn;

    if constexpr (reuse_a)
    {
        unpack_instrn = TT_OP_UNPACR0_TILE_INC(0, 1, BUF_DESC_ID_1, 1 /*Set Dvalid*/);
        // inc_l1_instrn = TT_OP_NOP;//TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 1);
        unpack_reuse_instrn = TT_OP_UNPACR1_TILE_INC(0, 0, BUF_DESC_ID_0, 1 /*Set Dvalid*/);
    }
    else
    {
        unpack_instrn = TT_OP_UNPACR1_TILE_INC(0, KT_DIM, BUF_DESC_ID_0, 1 /*Set Dvalid*/);
        // inc_l1_instrn = TT_OP_NOP;//TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, KT_DIM);
        unpack_reuse_instrn = TT_OP_UNPACR0_TILE_INC(0, 0, BUF_DESC_ID_1, 1 /*Set Dvalid*/);
    }
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_instrn /*, inc_l1_instrn*/);
    temp.set_start_op(unpack_reuse_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes unpacker to unpack operand 0 (BUF_DESC_ID_0) into SrcB
 * and unpacks operand 1 (BUF_DESC_ID_1) into SrcA. Matrix multiply FPU operation does SrcB * SrcA.
 * In order to get output of rowmajor matrix multiplication input 0 * Input 1, need to initialize
 * SrcA and SrcB to be input 1 & input 0 respectively.
 * The following matrix multiply has the following dimensions:
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim]
 * IMPORTANT NOTE:
 * This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]
 * kt_dim is assumed to be iterated over outside this api call
 * ct_dim * rt_dim <= 8 tiles in Float16b, ct_dim * rt_dim <= 4 tiles in Float32
 * @tparam BUF_DESC_ID_0/1: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam TRANSPOSE_EN: Enables transpose of a tile, currently only supported for SrcA,
 * but can support other unpackers
 * @tparam CT_DIM: number of tiles in the column dimension for input1 of matrix multiply
 * @tparam RT_DIM: number of tiles in the row dimension for input0 of matrix multiply
 * @tparam KT_DIM: number of tiles in the common dimension between input0 & input1 of matrix multiply
 */
template <uint32_t BUF_DESC_ID_0, uint32_t BUF_DESC_ID_1, bool TRANSPOSE_EN, std::uint8_t CT_DIM, std::uint8_t RT_DIM, uint32_t KT_DIM>
inline void _llk_unpack_matmul_init_()
{
    static_assert((TRANSPOSE_EN == false), "TODO: Transpose srcA not available yet");
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, TRANSPOSE_EN);

    _llk_unpack_matmul_mop_config_<BUF_DESC_ID_0, BUF_DESC_ID_1, CT_DIM, RT_DIM, KT_DIM>();
}

/**
 * @brief Performs unpack operation for matrix multiply such that:
 * Input 0 -> unpack to SrcB
 * Input 1 -> unpack to SrcA
 * Performs unpack for rt & ct dims of input0 & input 1 respectively
 * The following matrix multiply has the following dimensions:
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim]
 * IMPORTANT NOTE:
 * This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]
 * kt_dim is assumed to be iterated over outside this api call
 * ct_dim * rt_dim <= 8 tiles in Float16b, ct_dim * rt_dim <= 4 tiles in Float32
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer
 * start_l1_tile_idx_0 -> UNPACKER1 -> SRCB
 * start_l1_tile_idx_1 -> UNPACKER0 -> SRCA
 * @tparam CT_DIM: number of tiles in the column dimension for input1 of matrix multiply
 * @tparam RT_DIM: number of tiles in the row dimension for input0 of matrix multiply
 * @tparam KT_DIM: number of tiles in the common dimension between input0 & input1 of matrix multiply
 */
template <uint32_t BUF_DESC_ID_0, uint32_t BUF_DESC_ID_1, std::uint8_t CT_DIM, std::uint8_t RT_DIM, uint32_t KT_DIM>
inline void _llk_unpack_matmul_(const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1)
{
    // Reset Dest counters for Unpacker to 0
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    constexpr bool reuse_a        = CT_DIM >= RT_DIM;
    constexpr std::uint32_t t_dim = reuse_a ? RT_DIM : CT_DIM;

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        std::uint32_t tile_idx_0 = start_l1_tile_idx_0 + (reuse_a ? (t * KT_DIM) : 0);
        std::uint32_t tile_idx_1 = start_l1_tile_idx_1 + (reuse_a ? (0) : (t));

        // Set Source counter to L1 base + offset
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, tile_idx_0);
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, tile_idx_1);

        // Runs MOP
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }
}
