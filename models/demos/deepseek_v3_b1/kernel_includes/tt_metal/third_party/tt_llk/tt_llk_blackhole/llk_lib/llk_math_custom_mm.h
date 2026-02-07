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

template <bool transpose, bool split_acc, bool dense_packing>
inline void custom_mm_configure_addrmod() {
    constexpr std::uint8_t ADDR_MOD_0_SRCA_INCR = transpose ? 32 : 16;
    constexpr std::uint8_t ADDR_MOD_1_SRCA_INCR = transpose ? (64 - 16) : 16;
    constexpr std::uint16_t ADDR_MOD_1_DEST_INCR = split_acc ? (1024 - 8) : (1024 - 16);
    constexpr std::uint8_t ADDR_MOD_2_DEST_INCR = dense_packing ? 32 : 64;

    addr_mod_t{
        .srca = {.incr = ADDR_MOD_0_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
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
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_4);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 32, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_5);

    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_6);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_7);
}

template <bool split_acc>
inline void custom_mm_configure_mop(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim) {
    const std::uint32_t replay_buf_len = operandB_face_r_dim == 8 ? 11 : 9;

    load_replay_buf(ckernel::math::replay_buf_offset, replay_buf_len, [operandB_face_r_dim] {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // 16
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0  (8  if split_acc)

        // Finalization phase
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);  // 16 (24 if split_acc)

        TTI_MOVD2B(0, 32, ADDR_MOD_5, p_movd2a::MOV_4_ROWS, 0 + 8);
        TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 16 + 8);

        // Move lower 4 rows if they exist
        if (operandB_face_r_dim == 8) {
            TTI_MOVD2B(0, 0 + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 0 + 8 + 4);
            TTI_MOVD2B(0, 16 + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 16 + 8 + 4);
        }

        TTI_ZEROSRC(0, 1, 0, 1);

        TTI_ELWADD(0, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_6, 0);
        TTI_ELWADD(1, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
    });

    const std::uint32_t mvmul_base = lltt::replay_insn(ckernel::math::replay_buf_offset + 0, 3);
    const std::uint32_t mvmul_end_tile = TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
    const std::uint32_t mvmul_end_block = TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);

    ckernel_template tmp = ckernel_template(1, ct_dim, mvmul_base, mvmul_end_tile);
    tmp.set_last_inner_loop_instr(mvmul_end_block);
    tmp.set_last_outer_loop_instr(mvmul_end_block);

    tmp.program();
}

template <bool transpose = false, bool split_acc = false, bool dense_packing = false>
inline void _llk_math_custom_mm_init_(const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim = 1) {
    custom_mm_configure_addrmod<transpose, split_acc, dense_packing>();
    custom_mm_configure_mop<split_acc>(operandB_face_r_dim, ct_dim);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool finalize = true>
inline void _llk_math_custom_mm_(
    const std::uint32_t operandB_face_r_dim,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    const std::uint32_t replay_buf_len = operandB_face_r_dim == 8 ? 11 : 9;
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    for (std::uint32_t i = 0; i < kt_dim - 1; i++) {
        TTI_MOP(1, 0, 0);
    }

    if constexpr (finalize) {
        for (std::uint32_t i = 0; i < ct_dim - 1; i++) {
            lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
        }
        lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len - 1);
        TTI_ELWADD(3, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
    } else {
        TTI_MOP(1, 0, 0);
    }
}
