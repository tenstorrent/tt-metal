// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"

using namespace ckernel;

template <EltwiseBinaryType eltwise_binary_type, uint32_t num_tiles, MathFidelity math_fidelity>
inline void sdpa_bcast_col_srca_srcb_reuse_configure_mop(
    const std::uint32_t num_faces = 4, const std::uint32_t acc_to_dest = 0) {
    LLK_ASSERT(num_faces == 2, "num_faces must be 2");
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    constexpr uint addr_mod = ADDR_MOD_0;
    uint innerloop = 16 >> 3;  // 8 rows per eltwise op at a time.
    uint outerloop = num_faces;
    constexpr auto broadcast_type = p_elwise::SRCB_BCAST_COL;

    // Scalar broadcast should not Clear B within a mop.  This is controlled outside of MOP.
    load_replay_buf<NoExec>(ckernel::math::replay_buf_offset, 2, [] {
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 0, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 0);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 4, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 4);
    });

    uint d2a_instr = lltt::replay_insn(ckernel::math::replay_buf_offset, 2);
    if constexpr (eltwise_binary_type == ELWADD) {
        ckernel_template tmp(
            1, num_faces, d2a_instr, TT_OP_ELWADD(p_setrwc::CLR_NONE, acc_to_dest, broadcast_type, ADDR_MOD_0, 0));
        tmp.set_last_outer_loop_instr(TT_OP_ELWADD(p_setrwc::CLR_NONE, acc_to_dest, broadcast_type, ADDR_MOD_2, 0));
        tmp.program();
    } else if constexpr (eltwise_binary_type == ELWSUB) {
        ckernel_template tmp(
            1, num_faces, d2a_instr, TT_OP_ELWSUB(p_setrwc::CLR_NONE, acc_to_dest, broadcast_type, ADDR_MOD_0, 0));
        tmp.set_last_outer_loop_instr(TT_OP_ELWSUB(p_setrwc::CLR_NONE, acc_to_dest, broadcast_type, ADDR_MOD_2, 0));
        tmp.program();
    } else if constexpr (eltwise_binary_type == ELWMUL) {
        if constexpr (high_fidelity) {
            ckernel_template tmp(
                num_faces,
                to_underlying(math_fidelity),
                TT_OP_ELWMUL(p_setrwc::CLR_NONE, 0, broadcast_type, ADDR_MOD_0, 0));
            tmp.set_start_op(d2a_instr);
            tmp.set_last_inner_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_NONE, 0, broadcast_type, ADDR_MOD_2, 0));
            tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_NONE, 0, broadcast_type, ADDR_MOD_3, 0));
            tmp.program();
        } else {
            ckernel_template tmp(
                1, num_faces, d2a_instr, TT_OP_ELWMUL(p_setrwc::CLR_NONE, 0, broadcast_type, ADDR_MOD_0, 0));
            tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_NONE, 0, broadcast_type, ADDR_MOD_2, 0));
            tmp.program();
        }
    }
}

template <DstSync DST, bool IS_FP32_MATH_DEST_EN, bool clear_dest = false>
inline void _llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble_(uint32_t isrc) {
    TTI_STALLWAIT(
        p_stall::STALL_MATH,
        p_stall::WAIT_SFPU | p_stall::SRCA_VLD |
            p_stall::SRCB_VLD);  // MOVD2B for a whole face assumes unpacker will set a dummy
                                 // data_valid, so we want to wait on that
    uint src_index = isrc + get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, src_index);
    TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
    if constexpr (clear_dest) {
        if constexpr (DST == DstSync::SyncFull) {
            TT_ZEROACC(p_zeroacc::CLR_ALL, IS_FP32_MATH_DEST_EN, 0, ADDR_MOD_1, 0);
        } else {
            static_assert(DST == DstSync::SyncHalf);
            TT_ZEROACC(p_zeroacc::CLR_HALF, IS_FP32_MATH_DEST_EN, 0, ADDR_MOD_1, dest_offset_id);
        }
    }
}

template <
    EltwiseBinaryType eltwise_binary_type,
    uint32_t num_tiles,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool clear_dest = false,
    bool skip_signalling = false,
    bool fused_signalling = false>
inline void _llk_math_sdpa_bcast_col_srca_srcb_reuse_(uint dst_index) {
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);

    if constexpr (!fused_signalling) {
        for (std::uint32_t tile_num = 0; tile_num < num_tiles; tile_num++) {
            ckernel_template::run();
            if constexpr (!skip_signalling) {
                t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU);
            }
        }
    } else {
        for (std::uint32_t tile_num = 0; tile_num < num_tiles; tile_num += 2) {
            ckernel_template::run();
            ckernel_template::run();
            if constexpr (!skip_signalling) {
                t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU);
            }
        }
    }
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity, std::uint32_t FIDELITY_INCREMENT>
inline void sdpa_bcast_col_srca_srcb_reuse_configure_addrmod(const std::uint32_t num_faces) {
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    // Use srcA for data movement
    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_1);

        if constexpr (eltwise_binary_type == ELWMUL && high_fidelity) {
            addr_mod_t{
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 0, .cr = 1},
                .fidelity = {.incr = FIDELITY_INCREMENT}}
                .set(ADDR_MOD_0);

            addr_mod_t{
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1},
                .fidelity = {.incr = 0, .clr = 1}}
                .set(ADDR_MOD_2);
            addr_mod_t{
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = static_cast<int16_t>(8), .clr = 0, .cr = 0, .c_to_cr = 1},
                .fidelity = {.incr = 0, .clr = 1}}
                .set(ADDR_MOD_3);
        } else {
            addr_mod_t{
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_0);
            addr_mod_t{
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = static_cast<int16_t>(8)},
            }
                .set(ADDR_MOD_2);
        }
    }
}

template <EltwiseBinaryType eltwise_binary_type, uint32_t num_tiles, MathFidelity math_fidelity>
inline void _llk_math_sdpa_bcast_col_srca_srcb_reuse_init_(
    const std::uint32_t num_faces, const std::uint32_t acc_to_dest) {
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr std::uint32_t math_fidelity_increment = 1;

    sdpa_bcast_col_srca_srcb_reuse_configure_addrmod<eltwise_binary_type, math_fidelity, math_fidelity_increment>(
        num_faces);

    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        sdpa_bcast_col_srca_srcb_reuse_configure_mop<eltwise_binary_type, num_tiles, math_fidelity>(
            num_faces, acc_to_dest);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}
