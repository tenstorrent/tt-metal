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
inline void reduce_configure_addrmod();

template <ReduceDim dim, int num_fidelity_phases>
inline void reduce_configure_mop();

template <PoolType type, ReduceDim dim, int MATH_FIDELITY_DESC = 0, bool is_fp32_dest_acc_en = false, bool is_int_fpu_en = false>
inline void _llk_math_reduce_(const uint dst_index) {

    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr bool HIGH_FIDELITY = MATH_FIDELITY_PHASES > 0;

    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    if constexpr (dim == ReduceDim::REDUCE_ROW) {
        // Transpose for each face in src A done at unpacker, and pool
        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        } else {
            if constexpr (HIGH_FIDELITY) {
                ckernel_template::run(instrn_buffer);
                TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            }
        }

        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        } else {
            if constexpr (HIGH_FIDELITY) {
                ckernel_template::run(instrn_buffer);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            }
        }

        // Workaround for tenstorrent/budabackend#1948
        if constexpr (is_int_fpu_en) {
            TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
            TTI_SFPLOAD(0, 4, ADDR_MOD_0, 0);
            TTI_SFPSTORE(0,5,ADDR_MOD_0,0);
            TTI_SFPLOAD(0, 4, ADDR_MOD_0, 2);
            TTI_SFPSTORE(0,5,ADDR_MOD_0,2);
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
            TTI_SETC16(FP16A_FORCE_Enable_ADDR32, 0x1);
        }

        // Move back to B and transpose
        // we avoid clobbering weights in src B by moving to rows 16 - 31
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
        /*
        if constexpr (is_fp32_dest_acc_en) {
            if (0 == (((uint)unpack_dst_format[0]>>2)&0x1)) { // fp32 to fp16_a conversion
                TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
                TTI_SFPLOAD(0, 0, 3, 0);
                TTI_SFP_STOCH_RND(0,0,0,0,0,8);
                TTI_SFPSTORE(0,1,3,0);
            }
        }
        */
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        // Note: transpose on src B on works on rows 16 - 31
        TTI_TRNSPSRCB;
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        if constexpr (is_int_fpu_en) {
            TTI_SETC16(FP16A_FORCE_Enable_ADDR32, 0x0);
        }

        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
        TTI_ZEROSRC(0, 1, 0, 1); // Clear src A
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);

        // Increment dest by 32 for next accumulation
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_BD);

        /////////////////////
        // Second Tile Row //
        /////////////////////

        // Transpose at unpacker and pool
        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        } else {
            if constexpr (HIGH_FIDELITY) {
                ckernel_template::run(instrn_buffer);
                TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            }
        }

        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        } else {
            if constexpr (HIGH_FIDELITY) {
                ckernel_template::run(instrn_buffer);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            }
        }
        // Workaround for tenstorrent/budabackend#1948
        if constexpr (is_int_fpu_en) {
            TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
            TTI_SFPLOAD(0, 4, ADDR_MOD_0, 0);
            TTI_SFPSTORE(0,5,ADDR_MOD_0,0);
            TTI_SFPLOAD(0, 4, ADDR_MOD_0, 2);
            TTI_SFPSTORE(0,5,ADDR_MOD_0,2);
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
            TTI_SETC16(FP16A_FORCE_Enable_ADDR32, 0x1);
        }

        // Move back to B and transpose
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
        /*
        if constexpr (is_fp32_dest_acc_en) {
            if (0 == (((uint)unpack_dst_format[0]>>2)&0x1)) { // fp32 to fp16_a conversion
                TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
                TTI_SFPLOAD(0, 0, 3, 0);
                TTI_SFP_STOCH_RND(0,0,0,0,0,8);
                TTI_SFPSTORE(0,1,3,0);
            }
        }
        */
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        // Note: transpose on src B on works on rows 16 - 31
        TTI_TRNSPSRCB;
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        if constexpr (is_int_fpu_en) {
            TTI_SETC16(FP16A_FORCE_Enable_ADDR32, 0x0);
        }

        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
        TTI_ZEROSRC(0, 1, 0, 1); // Clear src A
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);

        // Increment dest by 32 for next accumulation
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
    } else if constexpr (dim == ReduceDim::REDUCE_COL) {
        for (int row_tile = 0; row_tile < 2; row_tile++) {
            // Just pool
            if constexpr (type == PoolType::MAX) {
                TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            } else {
                if constexpr (HIGH_FIDELITY) {
                    ckernel_template::run(instrn_buffer);
                } else {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
                }
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);

            if constexpr (type == PoolType::MAX) {
                TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            } else {
                if constexpr (HIGH_FIDELITY) {
                    ckernel_template::run(instrn_buffer);
                } else {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
                }
            }
            // Reset Dest Counter
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AD);
        }

    } else if constexpr (dim == ReduceDim::REDUCE_SCALAR) {
        //fp32 dest unsupported with reduce scalar, must fix zeroacc
        static_assert(!is_fp32_dest_acc_en);
        for (int tile = 0; tile < 3; tile++) {
            // Wait and pool
            if constexpr (type == PoolType::MAX) {
                TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4);
            } else {
                if constexpr (HIGH_FIDELITY) {
                    ckernel_template::run(instrn_buffer);
                    TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
                } else {
                    TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4);
                }
            }
        }
        // Wait and pool
        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4);
        } else {
            if constexpr (HIGH_FIDELITY) {
                ckernel_template::run(instrn_buffer);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4);
            }
        }

        // Need row in dest as column in src A
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
        // copy over from dest to B and do transpose
        // use rows 16 - 31 in src B as scratch
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 4);
        TTI_GATESRCRST(0b1,0b1);
        TTI_TRNSPSRCB;
        // gate math instructions until src B has been updated
        TTI_GATESRCRST(0b1,0b1);
        // copy over all 16 rows from B to A
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET +  0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET +  0);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET +  4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET +  4);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET +  8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET +  8);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
        // gate math instructions until src A has been updated by MOV instructions
        TTI_GATESRCRST(0b1,0b1);
        // zero out scratch in dest
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_1, 4);

        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        } else {
            if constexpr (HIGH_FIDELITY) {
                for (int i = 0; i < MATH_FIDELITY_PHASES - 1; i++) {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0);
                }
            }
            TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
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
        .srca = {.incr = 0},
        .srcb = {.incr = 1},
        .dest = {.incr = 1},
    }.set(ADDR_MOD_1);

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);

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
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 4));
        tmp.set_last_inner_loop_instr(
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4));
        tmp.set_last_outer_loop_instr(
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4));
        tmp.program(instrn_buffer);
    } else {
        ckernel_template tmp(1, num_fidelity_phases,
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0));
        tmp.set_last_inner_loop_instr(
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0));
        tmp.set_last_outer_loop_instr(
            TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0));
        tmp.program(instrn_buffer);
    }
}

template <PoolType type, ReduceDim dim, int MATH_FIDELITY_DESC = 0>
inline void _llk_math_reduce_init_(const std::uint32_t within_face_16x16_transpose=0) { //within_face_16x16_transpose used for unpack, ignored by math

    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr bool HIGH_FIDELITY = MATH_FIDELITY_PHASES > 0;

    reduce_configure_addrmod<type, MATH_FIDELITY_DESC>();
    if constexpr (HIGH_FIDELITY) {
        reduce_configure_mop<dim, MATH_FIDELITY_PHASES>();
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}
