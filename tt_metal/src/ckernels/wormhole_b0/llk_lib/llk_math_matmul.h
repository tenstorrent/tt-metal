#pragma once
#include "llk_param_structs.h"

#include "ckernel_include.h"
#include "ckernel_template.h"

#include "cmath_common.h"
#include "llk_math_common.h"
#include "debug_print.h"

#ifndef HF
#define HF 0
#endif

using namespace ckernel;

// local function declarations
inline void matmul_configure_addrmod();
inline void matmul_configure_mop();

template <int NUM_FIDELITY_PHASES>
inline void llk_math_matmul(uint dst_index) {
    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);

    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);

    if constexpr (high_fidelity) {
        ckernel_template::run(instrn_buffer);
        ckernel_template::run(instrn_buffer);
        TTI_INCRWC(2, 0, 8, 0);
        TTI_INCRWC(2, 0, 8, 0);
        TTI_INCRWC(2, 0, 8, 0);
        TTI_INCRWC(2, 0, 8, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        ckernel_template::run(instrn_buffer);
        ckernel_template::run(instrn_buffer);  // Run 4 times
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
    } else {
        ckernel_template::run(instrn_buffer);
        ckernel_template::run(instrn_buffer);  // Run mop twice
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 0);
    }
}

template <int NUM_FIDELITY_PHASES>
inline void matmul_configure_addrmod() {
    // MVMUL does D = B*A

    // Inner Loop --> 32/8 = 4 times for the full 32x16 face
    // DEST -- 8 rows are calculated each time
    // SRCB -- 8 rows are needed
    // SRCA -- full 16x16 gets used -- hardware will pair cols of A with rows of B
    // D[8,16] = B[8,16] * A[16,16]
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);

    if constexpr (high_fidelity) {
        // Fidelity Loop
        // DEST -- CR on dest for next fidelity phase
        // SRCB -- CR on srcB for next fidelity phase
        // SRCA -- CR on srcA for next fidelity phase
        // Fidelity -- increment phase
        addr_mod_t{
            .srca = {.incr = 0, .clr = 0, .cr = 1},
            .srcb = {.incr = 0, .clr = 0, .cr = 1},
            .dest = {.incr = 0, .clr = 0, .cr = 1},
            .fidelity = {.incr = 1, .clr = 0}}
            .set(ADDR_MOD_1);

        // Last outer loop,
        // DEST -- keep incrementing
        // SRCB -- CR to either go back to beginning or to next 32x16
        // SRCA -- increment to next 16x16
        // Fidelity -- reset fidelity
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 1},
            .srcb = {.incr =  0, .clr = 0, .cr = 1},
            .dest = {.incr = 32, .clr = 0, .cr = 1},
            .fidelity = {.incr = 0, .clr = 1}}
            .set(ADDR_MOD_2);
    }
    else {
        // Last inner loop,
        // DEST -- keep incrementing
        // SRCB -- CR to either go back to beginning or to next 32x16
        // SRCA -- increment to next 16x16
        // Fidelity -- reset fidelity
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr =  0, .clr = 0, .cr = 1},
            .dest = {.incr = 32, .clr = 0, .cr = 1},
            .fidelity = {.incr = 0, .clr = 1}}
            .set(ADDR_MOD_2);

        // Last outer loop,
        // DEST -- clear and CR on dest for next 32x16 x 16x16 matmul to accumulate on top
        // SRCB -- increment to next 32x16
        // SRCA -- increment to next 16x16
        addr_mod_t{
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 32, .clr = 0, .cr = 1},
            .dest = {.incr = 0, .clr = 1, .cr = 1},
            .fidelity = {.incr = 0, .clr = 0}}
            .set(ADDR_MOD_3);
    }
}

template <int NUM_FIDELITY_PHASES>
inline void matmul_configure_mop(bool transpose) {
    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);

    constexpr uint32_t num_inner_loops = 32 >> 3; //8 rows produced per op

    if constexpr (high_fidelity) {
        ckernel_template tmp(NUM_FIDELITY_PHASES, num_inner_loops, TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0));
        tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0));
        tmp.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0));
        tmp.set_start_op(TT_OP_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD|p_stall::SRCB_VLD));
        tmp.program(instrn_buffer);
    } else {
        // performs a 32x16 x 16x32 matmul
        ckernel_template tmp(2, num_inner_loops, TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0));
        tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0));
        tmp.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0));
        tmp.set_start_op(TT_OP_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD|p_stall::SRCB_VLD));

        tmp.program(instrn_buffer);
    }
}

template <int NUM_FIDELITY_PHASES>
inline void llk_math_matmul_init(std::uint32_t transpose=0) {
    matmul_configure_addrmod<NUM_FIDELITY_PHASES>();
    matmul_configure_mop<NUM_FIDELITY_PHASES>(transpose != 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}
