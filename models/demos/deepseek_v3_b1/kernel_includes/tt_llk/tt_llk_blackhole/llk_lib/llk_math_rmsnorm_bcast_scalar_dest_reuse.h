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

template <EltwiseBinaryType eltwise_binary_type, uint32_t num_tiles, int NUM_FIDELITY_PHASES = 0>
inline void rmsnorm_bcast_scalar_dest_reuse_configure_mop(
    const std::uint32_t num_faces = 4, const std::uint32_t acc_to_dest = 0) {
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);
    constexpr uint addr_mod = ADDR_MOD_0;
    uint innerloop = 16 >> 3;  // 8 rows per eltwise op at a time.
    uint outerloop = num_faces;
    constexpr auto broadcast_type = p_elwise::SRCB_BCAST_ALL;

    // Scalar broadcast should not Clear B within a mop.  This is controlled outside of MOP.
    if constexpr (eltwise_binary_type == ELWADD) {
        ckernel_template tmp(
            num_tiles,
            num_faces,
            TT_OP_ELWADD(0, acc_to_dest, broadcast_type, ADDR_MOD_0, 0),
            TT_OP_ELWADD(p_setrwc::CLR_A, acc_to_dest, broadcast_type, ADDR_MOD_2, 0));
        tmp.set_last_inner_loop_instr(TT_OP_ELWADD(p_setrwc::CLR_A, acc_to_dest, broadcast_type, ADDR_MOD_3, 0));
        tmp.program();
    } else if constexpr (eltwise_binary_type == ELWSUB) {
        ckernel_template tmp(
            num_tiles,
            num_faces,
            TT_OP_ELWSUB(0, acc_to_dest, broadcast_type, ADDR_MOD_0, 0),
            TT_OP_ELWSUB(p_setrwc::CLR_A, acc_to_dest, broadcast_type, ADDR_MOD_2, 0));
        tmp.set_last_inner_loop_instr(TT_OP_ELWSUB(p_setrwc::CLR_A, acc_to_dest, broadcast_type, ADDR_MOD_3, 0));
        tmp.program();
    } else if constexpr (eltwise_binary_type == ELWMUL) {
        if constexpr (high_fidelity) {
            ckernel_template tmp(
                num_faces,
                NUM_FIDELITY_PHASES,
                TT_OP_ELWMUL(0, 0, broadcast_type, ADDR_MOD_0, 0),
                TT_OP_ELWMUL(0, 0, broadcast_type, ADDR_MOD_2, 0));
            tmp.set_last_inner_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_3, 0));
            tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_4, 0));
            tmp.program();
        } else {
            ckernel_template tmp(
                num_tiles,
                num_faces,
                TT_OP_ELWMUL(0, 0, broadcast_type, ADDR_MOD_0, 0),
                TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_2, 0));
            tmp.set_last_inner_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_A, 0, broadcast_type, ADDR_MOD_3, 0));
            tmp.program();
        }
    }
}

inline void rmsnorm_bcast_scalar_reuse_dest_as_src() {
    TTI_STALLWAIT(
        p_stall::STALL_MATH,
        p_stall::WAIT_SFPU | p_stall::SRCB_VLD);  // MOVD2B for a whole face assumes unpacker will set a dummy
                                                  // data_valid, so we want to wait on that
    TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_1_ROW, 0);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    uint32_t num_tiles,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    int NUM_FIDELITY_PHASES = 0,
    bool clear_dest = false>
inline void _llk_math_rmsnorm_bcast_scalar_dest_reuse_(uint src_index, uint dst_index) {
    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);
    constexpr uint32_t ZERO_ACC_MODE = p_zeroacc::CLR_16;

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(src_index);
    rmsnorm_bcast_scalar_reuse_dest_as_src();
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB)) {
        ckernel_template::run();
    } else if constexpr (eltwise_binary_type == ELWMUL) {
        // Row and no broadcasted behaves similarly
        if constexpr (high_fidelity) {
#pragma GCC unroll 0
            for (std::uint32_t tile_num = 0; tile_num < num_tiles; tile_num++) {
                ckernel_template::run();
            }
        } else {
            ckernel_template::run();
        }
    }
    // Manually clear B once mop is done for scaler bcast
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);

    math::clear_dst_reg_addr();
}

template <EltwiseBinaryType eltwise_binary_type, int NUM_FIDELITY_PHASES, std::uint32_t FIDELITY_INCREMENT>
inline void rmsnorm_bcast_scalar_dest_reuse_configure_addrmod(const std::uint32_t num_faces) {
    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);
    // Use srcA for data movement
    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        addr_mod_t{
            .srca = {.incr = 8},
            .srcb = {.incr = 0},
            .dest = {.incr = 8},
        }
            .set(ADDR_MOD_0);
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_1);

        if constexpr (eltwise_binary_type == ELWMUL && high_fidelity) {
            addr_mod_t{
                .srca = {.incr = 0, .clr = 1},
                .srcb = {.incr = 0, .clr = 1},
                .dest = {.incr = 0, .clr = 0, .cr = 1},
                .fidelity = {.incr = FIDELITY_INCREMENT}}
                .set(ADDR_MOD_2);

            addr_mod_t{
                .srca = {.incr = 0, .clr = 1},
                .srcb = {.incr = 0, .clr = 1},
                .dest = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1},
                .fidelity = {.incr = 0, .clr = 1}}
                .set(ADDR_MOD_3);
            addr_mod_t{
                .srca = {.incr = 0, .clr = 1},
                .srcb = {.incr = 0, .clr = 1},
                .dest = {.incr = static_cast<int16_t>(8 + (4 - num_faces) * 16), .clr = 0, .cr = 0, .c_to_cr = 1},
                .fidelity = {.incr = 0, .clr = 1}}
                .set(ADDR_MOD_4);
        } else {
            addr_mod_t{
                .srca = {.incr = 0, .clr = 1},  // Clear SrcA counter to 0
                .srcb = {.incr = 0, .clr = 1},  // Clear SrcB counter to 0
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_2);
            addr_mod_t{
                .srca = {.incr = 0, .clr = 1},  // Clear SrcA counter to 0
                .srcb = {.incr = 0, .clr = 1},  // Clear SrcB counter to 0
                .dest = {.incr = static_cast<int16_t>(8 + (4 - num_faces) * 16)},
            }
                .set(ADDR_MOD_3);
        }
    }
}

template <EltwiseBinaryType eltwise_binary_type, uint32_t num_tiles, int MATH_FIDELITY_DESC = 0>
inline void _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_(
    const std::uint32_t num_faces, const std::uint32_t acc_to_dest) {
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr int MATH_FIDELITY_INCREMENT = get_math_fidelity_increment(MATH_FIDELITY_DESC);

    rmsnorm_bcast_scalar_dest_reuse_configure_addrmod<
        eltwise_binary_type,
        MATH_FIDELITY_PHASES,
        MATH_FIDELITY_INCREMENT>(num_faces);

    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        rmsnorm_bcast_scalar_dest_reuse_configure_mop<eltwise_binary_type, num_tiles, MATH_FIDELITY_PHASES>(
            num_faces, acc_to_dest);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}
