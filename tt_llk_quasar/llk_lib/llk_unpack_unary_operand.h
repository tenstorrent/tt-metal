// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @brief MOP configuration for unpack of unary operations
 * @details Sets up MOP for unpacking a single operand by tiles
 * works for any unpack resource
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32-bit mode
 * @param num_tiles: number of tiles to unpack at a time for a single operand, default 1 tile of 32x32
 */
template <uint32_t UNP_SEL, uint32_t BUF_DESC_ID, bool IS_32b_DEST_EN>
inline void _llk_unpack_unary_operand_mop_config_(const uint32_t num_tiles)
{
    static_assert(
        (UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B) || (UNP_SEL == p_unpacr::UNP_DEST),
        "UNP_SEL can only be set to p_unpacr::UNP_A/UNP_B/UNP_DEST");
    static_assert((BUF_DESC_ID < 16 && BUF_DESC_ID >= 0), "BUF_DESC_ID should be between 0-16 for unpackers");

    const uint32_t MOP_OUTER_LOOP     = num_tiles;
    constexpr uint32_t MOP_INNER_LOOP = 1;

    // RT: Use defines to remove these constexpr, and replace with a single TT_OP_UNPACR_FACE_INC
    constexpr static uint unpack_tile_instrn = []() constexpr
    {
        if constexpr (UNP_SEL == p_unpacr::UNP_A)
        {
            return TT_OP_UNPACR0_TILE_INC(0, 1 /*Src Tile Idx*/, BUF_DESC_ID, 1 /*Set Dvalid*/);
        }
        else if constexpr (UNP_SEL == p_unpacr::UNP_B)
        {
            return TT_OP_UNPACR1_TILE_INC(0, 1 /*Src Tile Idx*/, BUF_DESC_ID, 1 /*Set Dvalid*/);
        }
        else if constexpr (UNP_SEL == p_unpacr::UNP_DEST)
        {
            return TT_OP_UNPACR_DEST_TILE_INC(1, 1 /*Src Tile Idx*/, BUF_DESC_ID, 0 /*Set Dvalid*/);
        }
    }();

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);

    // If IS_32b_DEST_EN and UNP_SEL = UNP_A, zero out the SRCB reg
    // The only test in which there is a unary upk to SRCA with 32b DF is the datacopy kernel, which uses ELWADD
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
 * @brief MOP configuration for unpack to SrcA or SrcB with a tile transpose, implements input A -> A^T or B -> B^T
 * @tparam UNP_SEL: Selects which unpacker resource to use, supports p_unpacr::UNP_A or p_unpacr::UNP_B
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32-bit mode
 * @param num_tiles: number of tiles to unpack at a time for a single operand, default 1 tile of 32x32
 */
template <uint32_t UNP_SEL, uint32_t BUF_DESC_ID, bool IS_32b_DEST_EN>
inline void _llk_unpack_unary_operand_transpose_mop_config_(const uint32_t num_tiles)
{
    static_assert((UNP_SEL == p_unpacr::UNP_A) || (UNP_SEL == p_unpacr::UNP_B), "UNP_SEL can only be p_unpacr::UNP_A or p_unpacr::UNP_B for unpack transpose");
    static_assert((BUF_DESC_ID < 16 && BUF_DESC_ID >= 0), "BUF_DESC_ID should be between 0-16 for unpackers");

    const uint32_t MOP_OUTER_LOOP = num_tiles;
    const uint32_t MOP_INNER_LOOP = 1;

    constexpr uint replay_buf_len = NUM_FACES;

    load_replay_buf<0, replay_buf_len>(
        []
        {
            if constexpr (UNP_SEL == p_unpacr::UNP_A)
            {
                TTI_UNPACR0_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, BUF_DESC_ID, 0);                   // Unpacks face 0 into dest offset 0
                TTI_UNPACR0_FACE(1 /*Dst Face Idx*/, 2 /*Src Face Idx*/, 0, 0, BUF_DESC_ID, 0);                   // Unpacks face 2 into dest offset 1
                TTI_UNPACR0_FACE(2 /*Dst Face Idx*/, 1 /*Src Face Idx*/, 0, 0, BUF_DESC_ID, 0);                   // Unpacks face 1 into dest offset 2
                TTI_UNPACR0_FACE(3 /*Dst Face Idx*/, 3 /*Src Face Idx*/, 0, 0, BUF_DESC_ID, 1 /*Set DataValid*/); // Unpacks face 3 into dest offset 3
            }
            else if constexpr (UNP_SEL == p_unpacr::UNP_B)
            {
                TTI_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, BUF_DESC_ID, 0);
                TTI_UNPACR1_FACE(1 /*Dst Face Idx*/, 2 /*Src Face Idx*/, 0, 0, BUF_DESC_ID, 0);
                TTI_UNPACR1_FACE(2 /*Dst Face Idx*/, 1 /*Src Face Idx*/, 0, 0, BUF_DESC_ID, 0);
                TTI_UNPACR1_FACE(3 /*Dst Face Idx*/, 3 /*Src Face Idx*/, 0, 0, BUF_DESC_ID, 1 /*Set DataValid*/);
            }
        });
    ckernel_template temp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0),
        TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 1)); // Inc Src by 1 tile, because above UNPACR0/1_FACE do not inc counters

    // 32-bit datacopy uses ELWADD, which requires datavalid from both SrcA and SrcB
    if constexpr (IS_32b_DEST_EN)
    {
        if constexpr (UNP_SEL == p_unpacr::UNP_A)
        {
            temp.set_end_op(TT_OP_UNPACR_NOP(p_unpacr::UNP_B, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/));
        }
        else if constexpr (UNP_SEL == p_unpacr::UNP_B)
        {
            temp.set_end_op(TT_OP_UNPACR_NOP(p_unpacr::UNP_A, 1 /*Dvalid*/, 0, 0, 0 /*clear to 0*/, 0 /*clear to 0*/));
        }
    }

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialized unpacker to unpack a single operand by tiles
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @tparam BUF_DESC_ID: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0 - 16
 * @tparam TRANSPOSE_EN: Enables transpose of a tile, supported for SrcA and SrcB
 * @tparam IS_32b_DEST_EN: Set to True to enable using Math destination Register in 32-bit mode
 * @param num_tiles: number of tiles to unpack at a time for a single operand, default 1 tile of 32x32
 */
template <uint32_t UNP_SEL, uint32_t BUF_DESC_ID, bool TRANSPOSE_EN, bool IS_32b_DEST_EN>
inline void _llk_unpack_unary_operand_init_(const uint32_t num_tiles)
{
    if constexpr (UNP_SEL == p_unpacr::UNP_A || UNP_SEL == p_unpacr::UNP_DEST)
    {
        cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, TRANSPOSE_EN);
    }
    else if constexpr (UNP_SEL == p_unpacr::UNP_B)
    {
        cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, TRANSPOSE_EN);
    }

    if constexpr (TRANSPOSE_EN)
    {
        _llk_unpack_unary_operand_transpose_mop_config_<UNP_SEL, BUF_DESC_ID, IS_32b_DEST_EN>(num_tiles);
    }
    else
    {
        _llk_unpack_unary_operand_mop_config_<UNP_SEL, BUF_DESC_ID, IS_32b_DEST_EN>(num_tiles);
    }
}

/**
 * @brief Unpacks a single operand, works for any unpack resource
 * @tparam UNP_SEL: Selects which unpacker resource to use,
 * values = p_unpacr::UNP_A/p_unpacr::UNP_B/p_unpacr::UNP_DEST
 * @param l1_tile_idx: Index into the L1 buffer for a tile
 */
template <uint32_t UNP_SEL>
inline void _llk_unpack_unary_operand_(const uint l1_tile_idx)
{
    // RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users input offset idx

    // Reset Dest counters for Unpacker to 0
    // Set Source counter to L1 base + offset
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL == p_unpacr::UNP_DEST ? p_unpacr::UNP_A : UNP_SEL, l1_tile_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL == p_unpacr::UNP_DEST ? p_unpacr::UNP_A : UNP_SEL, 0);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
