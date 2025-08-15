// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/**
 * @brief Sets up mop config for eltwise unary datacopy operations
 * @tparam DATA_COPY_TYPE sets which src register to datacopy from values = <A2D, B2D>
 * @tparam IS_32b_DEST_EN set if math destination register is set to Float32/Int32 mode
 * @param num_rows_inner_loop: Number of rows that will be output by FPU per innerloop
 * Innerloop runs continuously without setting datavalids, or resetting counters.
 * If unpacker is unpacking 1 32x32 tile, with 1 dvalid -> set this value to 64 rows
 * If unpacker is unpacking 4 faces (16x16 each), with 4 dvalids -> set this value to 16 rows
 * @param num_dvalids_outer_loop: number of times required to reset datavalids for the unpacker
 * & counters for math srca/srcb.
 * If unpacker is unpacking 1 32x32 tile, with 1 dvalid -> set this value to 1
 * If unpacker is unpacking 4 faces (16x16 each), with 4 dvalids -> set this value to 4
 */
template <DataCopyType DATA_COPY_TYPE, bool IS_32b_DEST_EN>
inline void _llk_math_eltwise_unary_datacopy_mop_config_(
    const uint32_t num_rows_inner_loop, const uint32_t num_dvalids_outer_loop, const uint32_t num_rows_per_move_instrn)
{
    // Divide number of rows by how many rows are output per fpu instruction
    const uint32_t MOP_INNER_LOOP = num_rows_inner_loop >> math_rows_log2(num_rows_per_move_instrn);
    const uint32_t mov_rows_instn = (num_rows_per_move_instrn == 8)
                                        ? p_mov_src_to_dest::MOV_8_ROWS
                                        : ((num_rows_per_move_instrn == 4) ? p_mov_src_to_dest::MOV_4_ROWS : p_mov_src_to_dest::MOV_1_ROW);

    const uint32_t MOP_OUTER_LOOP = num_dvalids_outer_loop;

    const auto datacopy_func = [mov_rows_instn](uint8_t addr_mod)
    {
        if constexpr (IS_32b_DEST_EN)
        {
            return TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, addr_mod, 0);
        }
        else if constexpr (DATA_COPY_TYPE == DataCopyType::A2D)
        {
            return TT_OP_MOVA2D(0, 0, addr_mod, mov_rows_instn, 0);
        }
        else
        {
            return TT_OP_MOVB2D(0, 0, addr_mod, mov_rows_instn, 0, 0);
        }
    };

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, datacopy_func(ADDR_MOD_0));

    constexpr uint32_t CLR_SRC_VLD = IS_32b_DEST_EN                        ? p_cleardvalid::CLR_SRCAB_VLD
                                     : DATA_COPY_TYPE == DataCopyType::A2D ? p_cleardvalid::CLR_SRCA_VLD
                                                                           : p_cleardvalid::CLR_SRCB_VLD;

    // clear srcA and srcB dvalid
    temp.set_end_op(TT_OP_CLEARDVALID(CLR_SRC_VLD, 0, 0, 0, 0, 0));

    temp.set_last_inner_loop_instr(datacopy_func(ADDR_MOD_1));

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up addrmods for eltwise unary datacopy operations
 * @tparam DATA_COPY_TYPE sets which src register to datacopy from values = <A2D, B2D>
 */
template <DataCopyType DATA_COPY_TYPE>
inline void _llk_math_eltwise_unary_datacopy_addrmod_(const uint num_rows_per_move_instrn)
{
    constexpr uint8_t use_srca  = (DATA_COPY_TYPE == DataCopyType::A2D);
    constexpr uint8_t use_srcb  = (DATA_COPY_TYPE == DataCopyType::B2D);
    const uint8_t num_rows_srca = use_srca ? num_rows_per_move_instrn : 0;
    const uint8_t num_rows_srcb = use_srcb ? num_rows_per_move_instrn : 0;
    const uint8_t num_rows_dest = num_rows_per_move_instrn;

    // Increment rows for src register that is used, inc dest rows
    addr_mod_t {
        .srca = {.incr = num_rows_srca},
        .srcb = {.incr = num_rows_srcb},
        .dest = {.incr = num_rows_dest},
    }
        .set(ADDR_MOD_0);

    // Reset src counter that has been used, inc dest
    addr_mod_t {
        .srca = {.clr = use_srca},
        .srcb = {.clr = use_srcb},
        .dest = {.incr = num_rows_dest},
    }
        .set(ADDR_MOD_1);
}

/**
 * @brief Sets up initialization for eltwise unary datacopy operations
 * @tparam DATA_COPY_TYPE sets which src register to datacopy from values = <A2D, B2D>
 * @tparam IS_32b_DEST_EN set if math destination register is set to Float32/Int32 mode
 * @param num_rows_per_matrix: Number of rows that will be output by FPU per innerloop
 * Innerloop runs continuously without setting datavalids, or resetting counters.
 * If unpacker is unpacking 1 32x32 tile, with 1 dvalid -> set this value to 64 rows
 * If unpacker is unpacking 4 faces (16x16 each), with 4 dvalids -> set this value to 16 rows
 * @param num_matrices: Number of times required to reset datavalids for the unpacker
 * & counters for math srca/srcb.
 * If unpacker is unpacking 1 32x32 tile, with 1 dvalid -> set this value to 1
 * If unpacker is unpacking 4 faces (16x16 each), with 4 dvalids -> set this value to 4
 */
template <DataCopyType DATA_COPY_TYPE, bool IS_32b_DEST_EN>
inline void _llk_math_eltwise_unary_datacopy_init_(const uint num_rows_per_matrix, const uint num_matrices)
{
    // MOVA2D/MOVB2D can move 1, 4 or 8 rows, need to check which
    // For Float32 or Integer dest, ELWADD will be used for rebiasing, can only move MATH_ROWS
    const uint num_rows_per_move_instrn = [num_rows_per_matrix]() -> const uint
    {
        if constexpr (IS_32b_DEST_EN)
        {
            return ELTWISE_MATH_ROWS;
        }
        else
        {
            for (uint mr : MOVE_MATH_ROWS)
            {
                if (_divisible_by_pow_two_(num_rows_per_matrix, mr))
                {
                    return mr;
                }
            }
            return 1;
        }
    }();

    _llk_math_eltwise_unary_datacopy_addrmod_<DATA_COPY_TYPE>(num_rows_per_move_instrn);
    _llk_math_eltwise_unary_datacopy_mop_config_<DATA_COPY_TYPE, IS_32b_DEST_EN>(num_rows_per_matrix, num_matrices, num_rows_per_move_instrn);

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Perform an eltwise unary datacopy operation
 * @param tile_idx: Tile index into the destination register.
 * If dest reg in float16 mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in full mode
 * If dest reg in float32 mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 */
template <uint32_t num_rows_per_tile>
inline void _llk_math_eltwise_unary_datacopy_(const uint32_t tile_idx)
{
    _set_dst_write_addr_by_rows_<num_rows_per_tile>(tile_idx);

    // Run MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
