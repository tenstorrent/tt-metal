#pragma once
#include "llk_param_structs.h"

#include "ckernel_include.h"
#include "ckernel_template.h"

#include "cmath_common.h"
#include "llk_math_common.h"

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
    if constexpr (NUM_FIDELITY_PHASES > 0) {
        ckernel_template::run(instrn_buffer);
        ckernel_template::run(instrn_buffer);
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
        ckernel_template::run(instrn_buffer);
        ckernel_template::run(instrn_buffer);  // Run 4 times
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
    } else {
        ckernel_template::run(instrn_buffer);
        ckernel_template::run(instrn_buffer);  // Run mop twice
    }
}

template <int NUM_FIDELITY_PHASES>
inline void matmul_configure_addrmod() {
    // MVMUL does D = B*A

    // Inner Loop --> 8 times for the full 16x32 face
    // DEST -- 4 rows are calculated each time
    // SRCB -- 4 rows are needed
    // SRCA -- full 16x16 gets used -- hardware will pair cols of A with rows of B
    // D[4,16] = B[4,16] * A[16,16]
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 4, .clr = 0, .cr = 0},
        .dest = {.incr = 4, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    if constexpr (NUM_FIDELITY_PHASES > 0) {
        // Fidelity Loop
        // DEST -- CR on dest for next fidelity phase
        // SRCB -- Go back to beginning of srcB + Clear B Dvalid
        // SRCA -- Clear A Dvalid
        // Fidelity -- increment phase
        addr_mod_t{
            .srca = {.incr = 0, .clr = 1, .cr = 0},
            .srcb = {.incr = 0, .clr = 1, .cr = 0},
            .dest = {.incr = 0, .clr = 0, .cr = 1},
            .fidelity = {.incr = 1, .clr = 0}}
            .set(ADDR_MOD_1);
    }
    // Last inner loop,
    // DEST -- keep incrementing
    // SRCB -- Go back to beginning of srcB
    // SRCA -- Clear A Dvalid
    // Fidelity -- reset fidelity
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 32, .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_2);

    if constexpr (NUM_FIDELITY_PHASES == 0) {
        // Last outer loop,
        // DEST -- CR on dest for next fidelity phase
        // SRCB -- Go back to beginning of srcB + Clear B Dvalid
        // SRCA -- Clear A Dvalid
        // Fidelity -- increment phase
        addr_mod_t{
            .srca = {.incr = 0, .clr = 1, .cr = 0},
            .srcb = {.incr = 0, .clr = 1, .cr = 0},
            .dest = {.incr = 0, .clr = 1, .cr = 1},
            .fidelity = {.incr = 0, .clr = 0}}
            .set(ADDR_MOD_3);
    }
}

template <int NUM_FIDELITY_PHASES>
inline void matmul_configure_mop(bool transpose) {
    if constexpr (NUM_FIDELITY_PHASES > 0) {
        ckernel_template tmp(NUM_FIDELITY_PHASES, 8, TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0));
        if (transpose) {
            tmp.set_start_op(TT_OP_TRNSPSRCA);
        } 
        tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0));
        tmp.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0));
        tmp.program(instrn_buffer);
    } else {
        // 4x16x16 * 2x32x16
        ckernel_template tmp(2, 8, TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0));
        if (transpose) {
            tmp.set_start_op(TT_OP_TRNSPSRCA);
        } 
        tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0));
        tmp.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0));
        tmp.program(instrn_buffer);
    }
}

template <int NUM_FIDELITY_PHASES>
inline void llk_math_matmul_init(std::uint32_t transpose=0) {
    matmul_configure_addrmod<NUM_FIDELITY_PHASES>();
    matmul_configure_mop<NUM_FIDELITY_PHASES>(transpose>0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}
