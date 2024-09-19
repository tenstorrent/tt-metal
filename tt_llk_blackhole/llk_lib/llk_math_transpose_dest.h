// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "ckernel_include.h"
#include "ckernel_template.h"

#include "cmath_common.h"
#include "llk_math_common.h"
#include "ckernel_globals.h"

using namespace ckernel;

// local function declarations
inline void transpose_dest_configure_addrmod();

inline void _llk_math_transpose_dest_(const std::uint32_t dst_index) {
    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);

    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);

    ckernel_template::run(instrn_buffer);

    TTI_REPLAY(20, 5, 0, 0);
    TTI_REPLAY(26, 4, 0, 0);

    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);

    math::clear_dst_reg_addr();
}

inline void transpose_dest_configure_addrmod() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 16},
    }.set(ADDR_MOD_0);

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }.set(ADDR_MOD_1);

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = -16},
    }.set(ADDR_MOD_2);
}

inline void transpose_dest_configure_mop() {
    load_replay_buf<16, 16, 0>([] {
        // A
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET +  0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS,  0 - 16);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET +  4, ADDR_MOD_1, p_movd2b::MOV_4_ROWS,  4 - 16);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET +  8, ADDR_MOD_1, p_movd2b::MOV_4_ROWS,  8 - 16);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12 - 16);

        // B
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET +  0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS,  0);
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET +  4, ADDR_MOD_1, p_movd2b::MOV_4_ROWS,  4);
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET +  8, ADDR_MOD_1, p_movd2b::MOV_4_ROWS,  8);
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET + 12, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
        
        // C
        TTI_TRNSPSRCB;

        // D
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET  + 32, ADDR_MOD_2, p_movd2b::MOV_1_ROW,   0); // throwaway to decrement dst

        // E
        TTI_MOVB2D(0, p_movd2b::SRC_ROW16_OFFSET +  0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS,  0);
        TTI_MOVB2D(0, p_movd2b::SRC_ROW16_OFFSET +  4, ADDR_MOD_1, p_movb2d::MOV_4_ROWS,  4);
        TTI_MOVB2D(0, p_movd2b::SRC_ROW16_OFFSET +  8, ADDR_MOD_1, p_movb2d::MOV_4_ROWS,  8);
        TTI_MOVB2D(0, p_movd2b::SRC_ROW16_OFFSET + 12, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12);

        //F
        TTI_MOVB2D(0, p_movd2b::SRC_ZERO_OFFSET +  0, ADDR_MOD_1, p_movb2d::MOV_4_ROWS,  0);
        TTI_MOVB2D(0, p_movd2b::SRC_ZERO_OFFSET +  4, ADDR_MOD_1, p_movb2d::MOV_4_ROWS,  4);
    });

    uint AF = TT_OP_REPLAY(16, 16, 0, 0);
    uint BC = TT_OP_REPLAY(20, 5, 0, 0);
    uint E = TT_OP_REPLAY(26, 4, 0, 0);
    uint X = TT_OP_MOVB2D(0, p_movd2b::SRC_ZERO_OFFSET +  8, ADDR_MOD_1, p_movb2d::MOV_4_ROWS,  8);
    uint Y = TT_OP_MOVB2D(0, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12);

    ckernel_template tmp(1, 2, E, BC);
    tmp.set_start_op(BC);
    tmp.set_last_outer_loop_instr(AF);
    tmp.set_end_ops(X, Y);
    tmp.program(instrn_buffer);
}

inline void _llk_math_transpose_dest_init_() {

    transpose_dest_configure_addrmod();

    transpose_dest_configure_mop();
}
