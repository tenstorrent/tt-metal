// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"

using namespace ckernel;

/**
 * Configures address modifiers for specialized reduce_max_row operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block/tile operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce LLK configuration.
 * Use the standard reduce configuration functions for general-purpose reduction operations.
 */
inline void reduce_max_row_configure_addrmod()
{
    // ADDR_MOD_0: Default mode used with pool operations (GMPOOL/GAPOOL)
    // No auto-increment on any counter - keeps all address pointers at their current positions
    // Used for initial pooling operations where we want explicit control over counter advancement
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_0);

    // ADDR_MOD_1: Face-to-face advancement mode used with GMPOOL operations
    // Increments SrcA by 16 rows to advance to the next face (F0->F1, F2->F3)
    // Clears SrcB counter to prepare for subsequent MOVD2B operations that write transposed results
    // This allows processing pairs of faces (F0+F1, then F2+F3) efficiently
    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    // ADDR_MOD_2: Sequential 4-row write mode used with MOVB2D operations
    // Increments DEST by 4 rows after each operation (writes to rows 0, 4, 8, 12 within a face)
    // This distributes the transposed 1x16 reduction result across the first face (rows 0-15)
    // by writing 4 rows to every 4th row, matching the tile face layout
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 4, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_2);

    // ADDR_MOD_3: Face-boundary jump mode used with final MOVB2D in each face
    // Increments DEST by 20 rows, jumping from row 12 to row 32 (= 12 + 4 + 16)
    // This transitions from the last row of the current face to the first row of the next face
    // Example: After writing rows 0,4,8,12 in F0, this jumps to row 32 to start F2
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 20, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_3);
}

/**
 * Configures MOP (Macro Operation) for block-based reduce_max_row operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce LLK MOP configuration.
 * Use the standard reduce MOP configuration with _llk_math_reduce_init_ for general-purpose reduction.
 */
template <uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_()
{
    // Constraint on the outerloop and innerloop dim
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");

    // See _llk_math_reduce_max_row_ for a full algorithm explanation
    // Put the following 15 instructions in a REPLAY buffer
    lltt::record(0, 15);

    // Two GMPOOLs to pool F0 and F1 (or F2 and F3) together
    TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
    TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);

    if constexpr (is_fp32_dest_acc_en)
    {
        // FP32 destination mode, need to move high and low 16 bits to SrcB and transpose separately
        constexpr int dest_32b_hi = 0;
        constexpr int dest_32b_lo = 1;

        // The following instructions are repeated for F0&F1 reduced and F2&F3 reduced
        // Move high 16 bits from DEST row 0 to SrcB rows 16 - 31 and transpose
        TTI_MOVD2B(dest_32b_hi, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        TTI_TRNSPSRCB;

        // Move high 16 bits back to Dest
        TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 4);
        TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 8);
        TTI_MOVB2D(dest_32b_hi, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12);

        // Move low 16 bits to SrcB rows 16 - 31 and transpose
        TTI_MOVD2B(dest_32b_lo, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        TTI_TRNSPSRCB;

        // Move low 16 bits from SrcB rows 16 - 31 to DEST rows 0, 4, 8, 12
        // ADDR_MOD_2 increments CR_D and Dest counter val by 4, so that's why DEST location is '0', not '0, 4, 8, 12'.
        TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
        // ADDR_MOD_3 increments CR_D and Dest counter val by 20, to point to F2.
        TTI_MOVB2D(dest_32b_lo, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 0);
        // Clear B valid bits at the end and all address counters
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
    }
    else
    {
        // The following instructions are going to transpose the whole tile, unlike the FP32 mode.

        // Move row 0 from DEST to SrcB with offset of 16 rows and transpose
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        TTI_TRNSPSRCB;
        // Move row 0 from SrcB to DEST in 4-row chunks
        // ADDR_MOD_2 increments CR_D and Dest counter val by 4, so that's why DEST location is '0', not '0, 4, 8, 12'.
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
        // ADDR_MOD_3 increments CR_D and Dest counter val by 20, to point to F2.
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 0);

        // Move row 32 (F2R0) from DEST to SrcB with offset of 16 rows and transpose
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        TTI_TRNSPSRCB;
        // Move row 32 from SrcB to DEST in 4-row chunks
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 4);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 8);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12);

        // Clear B valid bits at the end and all address counters
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
    }

    static constexpr uint outer_loop = block_ct_dim;
    static constexpr uint inner_loop = 4;

    // Reduce F0 and F1 column-wise, doesn't change DEST counter
    static constexpr uint start_op = TT_OP_REPLAY(0, 2, 0, 0);
    // Increment DEST counter by 32 to point to F2 (DEST row 32)
    static constexpr uint inner_loop_op = TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    // Reduce F2 and F3 column-wise, doesn't change DEST counter, but clears A valid bits at the end
    static constexpr uint end_op_1 = TT_OP_REPLAY(0, 2, 0, 0);
    // Clear A valid bit to get another operand tile (scaler face stays the same)
    static constexpr uint end_op_2 = TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);

    ckernel::ckernel_template mop_template(outer_loop, inner_loop, inner_loop_op);
    mop_template.set_start_op(start_op);
    mop_template.set_end_ops(end_op_1, end_op_2);
    mop_template.program();
}

/**
 * Initializes block-based reduce_max_row operation for processing multiple tiles.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_math_reduce_init_ LLK.
 * Use the standard _llk_math_reduce_init_<PoolType::MAX, ReduceDim::REDUCE_ROW>() with multiple
 * _llk_math_reduce_() calls in a loop for general-purpose block reduction.
 */
template <uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_()
{
    reduce_max_row_configure_addrmod();

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);

    _llk_math_reduce_block_max_row_mop_config_<block_ct_dim, is_fp32_dest_acc_en>();
}

/**
 * Performs block-based reduce_max_row operation across multiple tiles in the width dimension.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_math_reduce_ LLK.
 * Use the standard _llk_math_reduce_<PoolType::MAX, ReduceDim::REDUCE_ROW>() in a loop
 * for general-purpose block reduction across multiple tiles.
 */
template <uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_(const uint dst_index)
{
    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);

    if constexpr (is_fp32_dest_acc_en)
    {
        // Run the MOP, performing a column reduce across all 4 faces
        ckernel::ckernel_template::run();
        // needs to be disabled for MOVD2B/B2D on BH (Issue ##449)
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
        // Replay the 12 instructions to transpose the reduced F0&F1 results
        lltt::replay(2, 12);
        // Replay the 13 instructions to transpose the reduced F2&F3 results
        // 13th instruction clears B valid bit to release SrcB bank and clears all address counters
        lltt::replay(2, 13);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
    }
    else
    {
        // Run the MOP, performing a column reduce across all 4 faces
        ckernel::ckernel_template::run();
        // Replay the 13 instructions to transpose the reduced results
        lltt::replay(2, 13);
    }
}
