// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
inline void reduce_configure_addrmod();

template <ReduceDim dim, int num_fidelity_phases>
inline void reduce_configure_mop();

template <PoolType type, ReduceDim dim, int MATH_FIDELITY_DESC = 0, bool is_fp32_dest_acc_en = false /* unused */, bool is_int_fpu_en = false /* unused */>
inline void _llk_math_reduce_(uint dst_index) {

    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr bool HIGH_FIDELITY = MATH_FIDELITY_PHASES > 0;

    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    if constexpr (dim == ReduceDim::REDUCE_ROW) {
        // Transpose and pool
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        TTI_TRNSPSRCA;
        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
        } else {
            if constexpr (HIGH_FIDELITY) {
                ckernel_template::run(instrn_buffer);
                TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
            }
        }
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        TTI_TRNSPSRCA;
        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
        } else {
            if constexpr (HIGH_FIDELITY) {
                ckernel_template::run(instrn_buffer);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
            }
        }

        // Move back to A and transpose
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_A);
        TTI_MOVD2A(p_movd2a::MOV_1_ROW, ADDR_MOD_2, 4, 0);  // Skip halo rows of A
        TTI_TRNSPSRCA;

        // Only move rows 2-15 back to dest, so no need to change counters
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);

        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);

        TTI_MOVA2D(
            p_mova2d::MOV_8_ROWS,
            ADDR_MOD_1,
            4,
            0);  // Switch to moving 8 rows for speed up, since no longer need row by row offset

        // Increment dest by 32 for next accumulation
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_AD);

        /////////////////////
        // Second Tile Row //
        /////////////////////

        // Transpose and pool
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        TTI_TRNSPSRCA;
        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
        } else {
            if constexpr (HIGH_FIDELITY) {
                ckernel_template::run(instrn_buffer);
                TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
            }
        }
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        TTI_TRNSPSRCA;
        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
        } else {
            if constexpr (HIGH_FIDELITY) {
                ckernel_template::run(instrn_buffer);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
            }
        }

        // Move back to A and transpose
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_A);
        TTI_MOVD2A(p_movd2a::MOV_1_ROW, ADDR_MOD_2, 4, 0);  // Skip halo rows of A
        TTI_TRNSPSRCA;

        // Only move rows 2-15 back to dest, so no need to change counters
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);

        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);
        TTI_MOVA2D(p_mova2d::MOV_1_ROW, ADDR_MOD_2, 4, 0);

        TTI_MOVA2D(
            p_mova2d::MOV_8_ROWS,
            ADDR_MOD_1,
            4,
            0);  // Switch to moving 8 rows for speed up, since no longer need row by row offset

        // Increment dest by 32 for next accumulation
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AD);
    } else if constexpr (dim == ReduceDim::REDUCE_COL) {
        for (int row_tile = 0; row_tile < 2; row_tile++) {
            // Transpose and pool
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);

            if constexpr (type == PoolType::MAX) {
                TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
            } else {
                if constexpr (HIGH_FIDELITY) {
                    ckernel_template::run(instrn_buffer);
                } else {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
                }
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);

            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
            if constexpr (type == PoolType::MAX) {
                TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
            } else {
                if constexpr (HIGH_FIDELITY) {
                    ckernel_template::run(instrn_buffer);
                } else {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
                }
            }
            // Reset Dest Counter
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AD);
        }
    } else if constexpr (dim == ReduceDim::REDUCE_SCALAR) {
        for (int tile = 0; tile < 3; tile++) {
            // Wait and pool
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
            if constexpr (type == PoolType::MAX) {
                TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, 4);
            } else {
                if constexpr (HIGH_FIDELITY) {
                    ckernel_template::run(instrn_buffer);
                    TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
                } else {
                    TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, 4);
                }
            }
        }
        // Wait and pool
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 4);
        } else {
            if constexpr (HIGH_FIDELITY) {
                ckernel_template::run(instrn_buffer);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 4);
            }
        }
        // Move back to A and transpose
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_A);
        TTI_MOVD2A(p_movd2a::MOV_1_ROW, ADDR_MOD_0, 4, 4);  // Skip halo rows of A
        TTI_TRNSPSRCA;
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, ADDR_MOD_0, 4);
        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
        } else {
            if constexpr (HIGH_FIDELITY) {
                for (int i = 0; i < MATH_FIDELITY_PHASES - 1; i++) {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, 0);
                }
            }
            TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, 0);
        }
    }
}

template <PoolType type, int MATH_FIDELITY_DESC>
inline void reduce_configure_addrmod() {

    constexpr int NUM_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr int FIDELITY_INCREMENT = get_math_fidelity_increment(MATH_FIDELITY_DESC);
    constexpr bool HIGH_FIDELITY = NUM_FIDELITY_PHASES > 0;

    addr_mod_t{
        .srca = {.incr = 0 },
        .srcb = {.incr = 0 },
        .dest = {.incr = 0 },
        .fidelity = { .incr = 0, .clr = 1}
    }.set(ADDR_MOD_0);
    addr_mod_t{
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_1);
    addr_mod_t{
        .srca = {.incr = 1},
        .srcb = {.incr = 0},
        .dest = {.incr = 1},
    }.set(ADDR_MOD_2);

    if constexpr (HIGH_FIDELITY) {
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
            .fidelity = { .incr = FIDELITY_INCREMENT}
        }.set(ADDR_MOD_3);
    }
}

template <ReduceDim dim, int num_fidelity_phases>
inline void reduce_configure_mop() {
    if constexpr (dim == ReduceDim::REDUCE_SCALAR) {
        ckernel_template tmp(1, num_fidelity_phases,
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, 4));
        tmp.set_last_inner_loop_instr(
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 4));
        tmp.set_last_outer_loop_instr(
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 4));
        tmp.program(instrn_buffer);
    } else {
        ckernel_template tmp(1, num_fidelity_phases,
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, 0));
        tmp.set_last_inner_loop_instr(
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 0));
        tmp.set_last_outer_loop_instr(
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, 0));
        tmp.program(instrn_buffer);
    }
}

template <PoolType type, ReduceDim dim, int MATH_FIDELITY_DESC = 0>
// within_face_16x16_transpose is used on WH but not used for GS, this transpose is done in math on GS
inline void _llk_math_reduce_init_(const std::uint32_t within_face_16x16_transpose=0) {

    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr bool HIGH_FIDELITY = MATH_FIDELITY_PHASES > 0;

    reduce_configure_addrmod<type, MATH_FIDELITY_DESC>();
    if constexpr (HIGH_FIDELITY) {
        reduce_configure_mop<dim, MATH_FIDELITY_PHASES>();
    }

    math::reset_counters(p_setrwc::SET_ABD_F);
}
