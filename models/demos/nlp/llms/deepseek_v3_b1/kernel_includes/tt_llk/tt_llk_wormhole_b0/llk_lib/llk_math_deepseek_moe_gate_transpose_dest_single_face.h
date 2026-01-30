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

namespace ckernel {

constexpr uint32_t transpose_dest_tile_offset = 64;  // 1 tile x 64 rows per tile

// Configure address modifiers for single face transpose
template <bool is_32bit>
inline void deepseek_moe_gate_transpose_dest_single_face_configure_addrmod() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = transpose_dest_tile_offset},
    }
        .set(ADDR_MOD_2);

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_3);
}

template <uint32_t num_tiles = 1, bool is_32bit>
inline void deepseek_moe_gate_transpose_dest_single_face_step0_configure_mop() {
    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    // For 16-bit data, simple single-pass transpose
    // Load replay buffer with the transpose sequence for one 16x16 face (face 0: rows 0-15)
    lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, 16);
    // Move 8 rows from DEST to SrcB interleaved (1 row at a time)
    TTI_MOVD2B(0, 16, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0);
    TTI_MOVD2B(0, 18, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 1);
    TTI_MOVD2B(0, 20, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 2);
    TTI_MOVD2B(0, 22, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 3);
    TTI_MOVD2B(0, 24, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 4);
    TTI_MOVD2B(0, 26, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 5);
    TTI_MOVD2B(0, 28, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 6);
    TTI_MOVD2B(0, 30, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 7);

    // Move 8 rows from SrcB back to DEST interleaved (1 row at a time)
    TTI_MOVB2D(0, 16, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 0);
    TTI_MOVB2D(0, 18, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 1);
    TTI_MOVB2D(0, 20, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 2);
    TTI_MOVB2D(0, 22, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 3);
    TTI_MOVB2D(0, 24, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 4);
    TTI_MOVB2D(0, 26, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 5);
    TTI_MOVB2D(0, 28, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 6);
    TTI_MOVB2D(0, 30, ADDR_MOD_2, p_movb2d::MOV_1_ROW, 7);

    uint d2b_instr = lltt::replay_insn(math::replay_buf_offset, 8);
    uint b2d_instr = lltt::replay_insn(math::replay_buf_offset + 8, 8);

    ckernel_template tmp(num_tiles, 1, d2b_instr, TT_OP_TRNSPSRCB);
    tmp.set_end_op(b2d_instr);
    tmp.program();
}

template <uint32_t num_tiles = 1, bool is_32bit>
inline void deepseek_moe_gate_transpose_dest_single_face_step1_configure_mop() {
    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    // For 16-bit data, simple single-pass transpose
    // Load replay buffer with the transpose sequence for one 16x16 face (face 0: rows 0-15)
    lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, 11);
    // Move 4 rows from DEST to SrcB (4 rows at a time)
    // This will place the first 2 rows in the first 2 columns, and the last 2 rows in the last 2 columns
    TTI_MOVD2B(0, 16, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 0);
    TTI_MOVD2B(0, 28, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 0);

    TTI_TRNSPSRCB;

    // Move 8 rows from SrcB back to DEST interleaved (1 row at a time)
    TTI_MOVB2D(0, 16, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 0);
    TTI_MOVB2D(0, 18, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 1);
    TTI_MOVB2D(0, 20, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 2);
    TTI_MOVB2D(0, 22, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 3);
    TTI_MOVB2D(0, 24, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 4);
    TTI_MOVB2D(0, 26, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 5);
    TTI_MOVB2D(0, 28, ADDR_MOD_3, p_movb2d::MOV_1_ROW, 6);
    TTI_MOVB2D(0, 30, ADDR_MOD_2, p_movb2d::MOV_1_ROW, 7);
    uint replay_instr = lltt::replay_insn(math::replay_buf_offset, 11);

    ckernel_template tmp(num_tiles, 1, replay_instr);
    tmp.program();
}

template <uint32_t num_tiles = 1, bool is_32bit>
inline void deepseek_moe_gate_transpose_dest_single_face_step2_configure_mop() {
    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    lltt::record<lltt::NoExec>(ckernel::math::replay_buf_offset, 4);
    // Move 8 rows from DEST to SrcB (4 rows at a time)
    TTI_MOVD2B(0, 16, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 0);
    TTI_MOVD2B(0, 20, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 4);

    TTI_TRNSPSRCB;

    // Move 1 row from SrcB back to DEST
    TTI_MOVB2D(0, 16, ADDR_MOD_2, p_movb2d::MOV_1_ROW, 0);
    uint replay_instr = lltt::replay_insn(math::replay_buf_offset, 4);

    ckernel_template tmp(num_tiles, 1, replay_instr);
    tmp.program();
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_deepseek_moe_gate_transpose_dest_single_face_common_init_() {
    deepseek_moe_gate_transpose_dest_single_face_configure_addrmod<is_32bit>();
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_deepseek_moe_gate_transpose_dest_single_face_step0_init_() {
    deepseek_moe_gate_transpose_dest_single_face_step0_configure_mop<4, is_32bit>();
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_deepseek_moe_gate_transpose_dest_single_face_step1_init_() {
    deepseek_moe_gate_transpose_dest_single_face_step1_configure_mop<3, is_32bit>();
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_deepseek_moe_gate_transpose_dest_single_face_step2_init_() {
    deepseek_moe_gate_transpose_dest_single_face_step2_configure_mop<2, is_32bit>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_deepseek_moe_gate_transpose_dest_single_face_step0_() {
    static_assert(
        !(is_32bit || is_fp32_dest_acc_en),
        "32-bit and fp32 dest accum enable are not supported for single face transpose");
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Wait for SFPU and SrcB to be available
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCB_VLD);

    // Run the 16-bit single-face transpose MOP
    ckernel_template::run();
}

// Perform in-place transpose on face 0 (rows 0-15) of a single tile in DEST
// dst_index: The tile index in DEST register buffer (0, 1, 2, ...)
//            The function transposes face 0 of the specified tile
template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_deepseek_moe_gate_transpose_dest_single_face_step1_() {
    static_assert(
        !(is_32bit || is_fp32_dest_acc_en),
        "32-bit and fp32 dest accum enable are not supported for single face transpose");

    math::reset_counters(p_setrwc::SET_ABD_F);

    // Wait for SFPU and SrcB to be available
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCB_VLD);

    // Run the 16-bit single-face transpose MOP
    ckernel_template::run();
}

// Perform in-place transpose on face 0 (rows 0-15) of a single tile in DEST
// dst_index: The tile index in DEST register buffer (0, 1, 2, ...)
//            The function transposes face 0 of the specified tile
template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_deepseek_moe_gate_transpose_dest_single_face_step2_() {
    static_assert(
        !(is_32bit || is_fp32_dest_acc_en),
        "32-bit and fp32 dest accum enable are not supported for single face transpose");

    math::reset_counters(p_setrwc::SET_ABD_F);

    // Wait for SFPU and SrcB to be available
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCB_VLD);

    ckernel_template::run();

    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

}  // namespace ckernel
