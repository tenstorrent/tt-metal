
#include "llk_param_structs.h"

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

template <PoolType type, ReduceDim dim, int num_fidelity_phases = 0, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce(uint dst_index) {
    constexpr bool high_fidelity = num_fidelity_phases > 0 && num_fidelity_phases <= 4;
    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    if constexpr (dim == ReduceDim::REDUCE_ROW) {
        // Transpose for each face in src A done at unpacker, and pool
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);

        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        } else {
            if constexpr (high_fidelity) {
                ckernel_template::run(instrn_buffer);
                TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            }
        }
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);

        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        } else {
            if constexpr (high_fidelity) {
                ckernel_template::run(instrn_buffer);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            }
        }

        // Move back to B and transpose
        // we avoid clobbering weights in src B by moving to rows 16 - 31
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_1, p_movd2b::MOV_1_ROW, 0);
        TTI_GATESRCRST(0b1,0b1);
        // Note: transpose on src B on works on rows 16 - 31
        TTI_TRNSPSRCB;
        // gate math instructions until src B has been updated
        TTI_GATESRCRST(0b1,0b1);
        // Only move rows 2-15 back to dest, so no need to change counters
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_1, p_movb2d::MOV_1_ROW, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_1, p_movb2d::MOV_1_ROW, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_1, p_movb2d::MOV_1_ROW, 0);

        // Switch to moving 4 rows for speed up, since no longer need row by row offset
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);

        // Increment dest by 32 for next accumulation
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_BD);

        /////////////////////
        // Second Tile Row //
        /////////////////////

        // Transpose at unpacker and pool
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);

        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        } else {
            if constexpr (high_fidelity) {
                ckernel_template::run(instrn_buffer);
                TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);                
            }
        }
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);

        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        } else {
            if constexpr (high_fidelity) {
                ckernel_template::run(instrn_buffer);
            } else {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            }
        }

        // Move back to B and transpose
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_1, p_movd2b::MOV_1_ROW, 0);
        TTI_GATESRCRST(0b1,0b1);
        // Note: transpose on src B on works on rows 16 - 31
        TTI_TRNSPSRCB;
        // gate math instructions until src B has been updated
        TTI_GATESRCRST(0b1,0b1);
        // Only move rows 2-15 back to dest, so no need to change counters
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_1, p_movb2d::MOV_1_ROW, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_1, p_movb2d::MOV_1_ROW, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_1, p_movb2d::MOV_1_ROW, 0);

        // Switch to moving 4 rows for speed up, since no longer need row by row offset
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);

        // Increment dest by 32 for next accumulation
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
    } else if constexpr (dim == ReduceDim::REDUCE_COL) {
        for (int row_tile = 0; row_tile < 2; row_tile++) {
            // Just pool
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);

            if constexpr (type == PoolType::MAX) {
                TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            } else {
                if constexpr (high_fidelity) {
                    ckernel_template::run(instrn_buffer);
                } else {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);                    
                }
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);

            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
            if constexpr (type == PoolType::MAX) {
                TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
            } else {
                if constexpr (high_fidelity) {
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
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
            if constexpr (type == PoolType::MAX) {
                TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4);
            } else {
                if constexpr (high_fidelity) {
                    ckernel_template::run(instrn_buffer);
                    TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
                } else {
                    TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4);
                }
            }
        }
        // Wait and pool
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4);
        } else {
            if constexpr (high_fidelity) {
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
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, ADDR_MOD_0, 4);

        if constexpr (type == PoolType::MAX) {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        } else {
            if constexpr (high_fidelity) {
                for (int i = 0; i < num_fidelity_phases - 1; i++) {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0);
                }
            }
            TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }
    }
}

template <PoolType type, bool is_high_fidelity>
inline void reduce_configure_addrmod() {
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
        .srcb = {.incr = 4},
        .dest = {.incr = 4},
    }
        .set(ADDR_MOD_2);

    if constexpr (is_high_fidelity) {
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
            .fidelity = { .incr = 1}
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

template <PoolType type, ReduceDim dim, int num_fidelity_phases = 0>
inline void llk_math_reduce_init(const std::uint32_t within_face_16x16_transpose=0) { //within_face_16x16_transpose used for unpack, ignored by math
    
    constexpr bool high_fidelity = num_fidelity_phases > 0 && num_fidelity_phases <= 4;
    
    reduce_configure_addrmod<type, high_fidelity>();
    if constexpr (high_fidelity) {
        reduce_configure_mop<dim, num_fidelity_phases>();
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0); 
    
    math::reset_counters(p_setrwc::SET_ABD_F);
}
