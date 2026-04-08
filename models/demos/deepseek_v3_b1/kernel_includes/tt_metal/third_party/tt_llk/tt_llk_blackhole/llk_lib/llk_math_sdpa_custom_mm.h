// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"

using namespace ckernel;
using namespace ckernel::math;

template <bool transpose>
inline void sdpa_custom_mm_configure_addrmod() {
    constexpr std::uint32_t face_r_dim = 8;
    constexpr std::uint8_t ADDR_MOD_0_SRCA_INCR = transpose ? 32 : 16;
    constexpr std::uint8_t ADDR_MOD_1_SRCA_INCR = transpose ? (64 - 16) : 16;
    constexpr std::uint16_t ADDR_MOD_1_DEST_INCR = (1024 - face_r_dim);
    constexpr std::uint8_t ADDR_MOD_2_DEST_INCR = 2 * face_r_dim;

    addr_mod_t{
        .srca = {.incr = ADDR_MOD_0_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = face_r_dim, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    addr_mod_t{
        .srca = {.incr = ADDR_MOD_1_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = ADDR_MOD_1_DEST_INCR, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = ADDR_MOD_2_DEST_INCR, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_2);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_3);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 1, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_4);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_7);
}

inline void sdpa_custom_mm_configure_mop(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim) {
    constexpr std::uint32_t replay_buf_len = 3;
    load_replay_buf(ckernel::math::replay_buf_offset, replay_buf_len, [operandB_face_r_dim] {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // 16
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
    });

    const std::uint32_t mvmul_base = lltt::replay_insn(ckernel::math::replay_buf_offset + 0, replay_buf_len);
    const std::uint32_t mvmul_end_tile = TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
    const std::uint32_t mvmul_end_block = TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);

    ckernel_template tmp = ckernel_template(1, ct_dim, mvmul_base, mvmul_end_tile);
    tmp.set_last_inner_loop_instr(mvmul_end_block);
    tmp.set_last_outer_loop_instr(mvmul_end_block);

    tmp.program();
}

template <bool transpose = false>
inline void _llk_math_sdpa_custom_mm_init_(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim = 1) {
    sdpa_custom_mm_configure_addrmod<transpose>();
    sdpa_custom_mm_configure_mop(operandB_face_r_dim, ct_dim);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_sdpa_custom_mm_mask_dest_(
    const std::uint32_t dst_index, const std::uint32_t ct_dim, bool mask_chunk = false) {
    // Zero Dest
    uint32_t dst_offset = dst_index + get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_offset);
    if (mask_chunk) {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD);
        // Zero Dest
        uint32_t dst_face = dst_offset / 16;
        for (uint32_t i = 0; i < ct_dim * 2; i++) {
            TTI_MOVB2D(0, p_movb2d::SRC_ZERO_OFFSET, ADDR_MOD_4, p_movb2d::MOV_8_ROW_BRCST, 0);
        }
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
    } else {
        // Zero Dest
        uint32_t dst_face = dst_offset / 16;
        // A tile is 2 faces of 8x16, so clearing 16 rows per tile is equivalent to clearing the tile
        for (uint32_t i = 0; i < ct_dim; i++) {
            TT_ZEROACC(p_zeroacc::CLR_16, 0, 0, ADDR_MOD_7, dst_face);
            dst_face++;
        }
    }
}

inline void _llk_math_sdpa_custom_mm_(
    const std::uint32_t operandB_face_r_dim,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1,
    const bool mask_chunk = false) {
    // dst offset initialized by _llk_math_sdpa_custom_mm_mask_dest_
    _llk_math_sdpa_custom_mm_mask_dest_(dst_index, ct_dim, mask_chunk);

    for (std::uint32_t i = 0; i < kt_dim - 1; i++) {
        TTI_MOP(1, 0, 0);
    }
    for (uint32_t i = 0; i < ct_dim - 1; i++) {
        lltt::replay(ckernel::math::replay_buf_offset, 3);
        TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
        t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU);
    }
    lltt::replay(ckernel::math::replay_buf_offset, 3);
    TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);
    t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU);
}
