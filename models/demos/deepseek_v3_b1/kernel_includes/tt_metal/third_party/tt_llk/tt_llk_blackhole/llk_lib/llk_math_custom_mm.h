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

template <bool transpose, bool split_acc>
inline void custom_mm_configure_addrmod() {
    constexpr std::uint8_t ADDR_MOD_0_SRCA_INCR = transpose ? 32 : 16;
    constexpr std::uint8_t ADDR_MOD_1_SRCA_INCR = transpose ? (64 - 16) : 16;
    constexpr std::uint16_t ADDR_MOD_1_DEST_INCR = split_acc ? (1024 - 8) : 0;
    constexpr std::uint8_t ADDR_MOD_1_DEST_CLR = split_acc ? 0 : 1;

    addr_mod_t{
        .srca = {.incr = ADDR_MOD_0_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    addr_mod_t{
        .srca = {.incr = ADDR_MOD_1_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = ADDR_MOD_1_DEST_INCR, .clr = ADDR_MOD_1_DEST_CLR, .cr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_3);
}

template <bool split_acc>
inline void custom_mm_configure_mop(const std::uint32_t operandB_face_r_dim) {
    const std::uint32_t replay_buf_len = operandB_face_r_dim == 8 ? 15 : 11;

    load_replay_buf(ckernel::math::replay_buf_offset, replay_buf_len, [operandB_face_r_dim] {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // 16
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // 0  (8  if split_acc)
        TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);    // 16 (24 if split_acc)

        // Finalization phase
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);  // 16 (24 if split_acc)

        TTI_MOVD2A(0, 0, ADDR_MOD_3, p_movd2a::MOV_4_ROWS, 0);
        TTI_MOVD2A(0, 16, ADDR_MOD_3, p_movd2a::MOV_4_ROWS, 16);

        // Move lower 4 rows if they exist
        if (operandB_face_r_dim == 8) {
            TTI_MOVD2A(0, 0 + 4, ADDR_MOD_3, p_movd2a::MOV_4_ROWS, 0 + 4);
            TTI_MOVD2A(0, 16 + 4, ADDR_MOD_3, p_movd2a::MOV_4_ROWS, 16 + 4);
        }

        TTI_MOVD2B(0, 0, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 0 + 8);
        TTI_MOVD2B(0, 16, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 16 + 8);

        // Move lower 4 rows if they exist
        if (operandB_face_r_dim == 8) {
            TTI_MOVD2B(0, 0 + 4, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 0 + 8 + 4);
            TTI_MOVD2B(0, 16 + 4, ADDR_MOD_3, p_movd2b::MOV_4_ROWS, 16 + 8 + 4);
        }

        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
        TTI_ELWADD(3, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
    });

    const std::uint32_t mvmul_base = lltt::replay_insn(ckernel::math::replay_buf_offset + 0, 3);
    const std::uint32_t mvmul_final = lltt::replay_insn(ckernel::math::replay_buf_offset + 3, 1);
    const std::uint32_t finalize = lltt::replay_insn(ckernel::math::replay_buf_offset + 4, replay_buf_len - 4);

    ckernel_unpack_template tmp = ckernel_unpack_template(
        true, false, mvmul_base, 0, 0, 0, mvmul_base, mvmul_final, split_acc ? finalize : mvmul_final);

    tmp.program();
}

template <bool transpose = false, bool split_acc = false>
inline void _llk_math_custom_mm_init_(const std::uint32_t operandB_face_r_dim) {
    custom_mm_configure_addrmod<transpose, split_acc>();
    custom_mm_configure_mop<split_acc>(operandB_face_r_dim);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool finalize = true>
inline void _llk_math_custom_mm_(const std::uint32_t dst_index, const std::uint32_t kt_dim = 1) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    for (uint32_t i = 0; i < kt_dim - 1; i++) {
        TTI_MOP(0, 0, 0);
    }

    if constexpr (finalize) {
        TTI_MOP(0, 0, 1);
    } else {
        TTI_MOP(0, 0, 0);
    }
}
