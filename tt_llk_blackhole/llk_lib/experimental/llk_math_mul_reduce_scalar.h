// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "llk_operands.h"

using namespace ckernel;

/*************************************************************************
 * LLK MUL REDUCE SCALAR - Low-level reduce operations for fused mul+reduce
 *************************************************************************/

// Helper macros for moving destination to source registers
#define MOVD2A_4_ROWS(base_offset, row_offset) \
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + (row_offset) * 4, ADDR_MOD_0, p_movd2a::MOV_4_ROWS, (base_offset) + (row_offset) * 4);

#define MOVD2A_16_ROWS(base_offset) \
    MOVD2A_4_ROWS(base_offset, 0)   \
    MOVD2A_4_ROWS(base_offset, 1)   \
    MOVD2A_4_ROWS(base_offset, 2)   \
    MOVD2A_4_ROWS(base_offset, 3)   \
    MOVD2A_4_ROWS(base_offset, 4)   \
    MOVD2A_4_ROWS(base_offset, 5)   \
    MOVD2A_4_ROWS(base_offset, 6)   \
    MOVD2A_4_ROWS(base_offset, 7)   \
    MOVD2A_4_ROWS(base_offset, 8)   \
    MOVD2A_4_ROWS(base_offset, 9)   \
    MOVD2A_4_ROWS(base_offset, 10)  \
    MOVD2A_4_ROWS(base_offset, 11)  \
    MOVD2A_4_ROWS(base_offset, 12)  \
    MOVD2A_4_ROWS(base_offset, 13)  \
    MOVD2A_4_ROWS(base_offset, 14)  \
    MOVD2A_4_ROWS(base_offset, 15)

/**
 * @brief Execute GAPOOL operations with optional high fidelity phases
 *
 * Executes the appropriate number of GAPOOL instructions based on fidelity level.
 * For high fidelity (MATH_FIDELITY_PHASES > 0), executes MATH_FIDELITY_PHASES - 1
 * iterations with ADDR_MOD_2, followed by one final GAPOOL with ADDR_MOD_0.
 *
 * @tparam MATH_FIDELITY_PHASES Number of fidelity phases (0, 2, 3, or 4)
 */
template <int MATH_FIDELITY_PHASES>
inline void execute_high_fidelity_gapool()
{
    static_assert(
        MATH_FIDELITY_PHASES == 0 || MATH_FIDELITY_PHASES == 2 || MATH_FIDELITY_PHASES == 3 || MATH_FIDELITY_PHASES == 4,
        "MATH_FIDELITY_PHASES must be 0, 2, 3, or 4");

    if constexpr (MATH_FIDELITY_PHASES >= 2)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
    }
    if constexpr (MATH_FIDELITY_PHASES >= 3)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
    }
    if constexpr (MATH_FIDELITY_PHASES >= 4)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
    }

    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
}

/**
 * @brief Move destination tile to source registers for mul_reduce_scalar
 *
 * Moves data from destination registers to source A or B registers.
 *
 * @tparam binary_reuse_dest Direction: DEST_TO_SRCA or DEST_TO_SRCB
 * @param idst Destination tile index (0-7)
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_mul_reduce_scalar_move_dest_to_src_(std::uint32_t idst = 0)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        if (idst == 0)
        {
            TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base());
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        }

        switch (idst)
        {
            case 0:
                MOVD2A_16_ROWS(0);
                break;
            case 1:
                MOVD2A_16_ROWS(64);
                break;
            case 2:
                MOVD2A_16_ROWS(128);
                break;
            case 3:
                MOVD2A_16_ROWS(192);
                break;
            case 4:
                MOVD2A_16_ROWS(256);
                break;
            case 5:
                MOVD2A_16_ROWS(320);
                break;
            case 6:
                MOVD2A_16_ROWS(384);
                break;
            case 7:
                MOVD2A_16_ROWS(448);
                break;
            default:
                break;
        }
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_BD);

        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 16, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 16);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 20, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 20);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 24, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 24);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 28, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 28);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 32, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 32);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 36, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 36);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 40, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 40);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 44, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 44);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 48, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 48);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 52, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 52);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 56, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 56);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 60, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 60);
    }
}

/**
 * @brief Configure address modifiers for mul_reduce_scalar operations
 *
 * Sets up address modifiers for source A, source B, destination, and fidelity
 * registers based on the specified math fidelity level.
 *
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 */
template <int MATH_FIDELITY_DESC>
inline void mul_reduce_scalar_configure_addrmod()
{
    constexpr int NUM_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr int FIDELITY_INCREMENT  = get_math_fidelity_increment(MATH_FIDELITY_DESC);
    constexpr bool HIGH_FIDELITY      = NUM_FIDELITY_PHASES > 0;

    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);
    addr_mod_t {.srca = {.incr = 16}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.clr = 1}}.set(ADDR_MOD_1);

    if constexpr (HIGH_FIDELITY)
    {
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = FIDELITY_INCREMENT}}.set(ADDR_MOD_2);
    }
}

/**
 * @brief Initialize reduce for mul_reduce_scalar operation
 *
 * Configures address modifiers and resets counters for the fused
 * multiply-reduce-scalar operation.
 *
 * @tparam is_fp32_dest_acc_en If true, enables FP32 destination accumulation
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 * @tparam enforce_fp32_accumulation If true, enforces FP32 accumulation (requires is_fp32_dest_acc_en)
 */
template <bool is_fp32_dest_acc_en, int MATH_FIDELITY_DESC = 0, bool enforce_fp32_accumulation = false>
inline void _llk_math_mul_reduce_scalar_init_()
{
    mul_reduce_scalar_configure_addrmod<MATH_FIDELITY_DESC>();

    if constexpr (enforce_fp32_accumulation)
    {
        static_assert(is_fp32_dest_acc_en, "FP32 Dest must be enabled for FP32 accumulation");
    }
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Perform column reduction for mul_reduce_scalar (accumulates across tiles)
 *
 * Performs column-wise reduction that accumulates results into the specified
 * destination tile. Used in a loop to accumulate multiple input tiles.
 *
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 * @param dst_index Destination tile index to accumulate into (0-7)
 * @param narrow_tile If true, process only 2 row tiles instead of full tile
 * @param num_faces Number of faces (1, 2, or 4)
 */
template <int MATH_FIDELITY_DESC = 0>
inline void _llk_math_mul_reduce_column_(const std::uint32_t dst_index, bool narrow_tile = false, const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    const std::uint32_t num_row_tiles  = narrow_tile ? 2 : ((num_faces > 1) ? num_faces / 2 : 1);

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    for (std::uint32_t row_tile = 0; row_tile < num_row_tiles; row_tile++)
    {
        execute_high_fidelity_gapool<MATH_FIDELITY_PHASES>();

        if ((!narrow_tile) && (num_faces > 1))
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_A, 0, 0, 8, p_setrwc::SET_A);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_A, 0, 0, 8, p_setrwc::SET_A);

            execute_high_fidelity_gapool<MATH_FIDELITY_PHASES>();
        }

        if (row_tile == 0)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_A, 0, 0, 8, p_setrwc::SET_A);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_A, 0, 0, 8, p_setrwc::SET_A);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        }
        else
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AD);
        }
    }
}

/**
 * @brief Perform final scalar reduction for mul_reduce_scalar
 *
 * Collapses a column-reduced tile into a single scalar value stored in the
 * first element of the output tile.
 *
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 */
template <int MATH_FIDELITY_DESC = 0>
inline void _llk_math_mul_reduce_scalar_()
{
    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);

    // Copy row 0 from dest to srcB (rows 16-31 as scratch) and transpose
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_GATESRCRST(0b1, 0b1);
    TTI_TRNSPSRCB;
    TTI_GATESRCRST(0b1, 0b1);

    // Copy all 16 rows from srcB to srcA
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
    TTI_GATESRCRST(0b1, 0b1);

    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 0);

    execute_high_fidelity_gapool<MATH_FIDELITY_PHASES>();
}

/**
 * @brief Clear data valid flags after mul_reduce_scalar operation
 *
 * Clears DVALID flags and resets all counters. Should be called after
 * mul_reduce_scalar operations are complete.
 */
inline void _llk_math_mul_reduce_scalar_clear_dvalid_()
{
    TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
}
