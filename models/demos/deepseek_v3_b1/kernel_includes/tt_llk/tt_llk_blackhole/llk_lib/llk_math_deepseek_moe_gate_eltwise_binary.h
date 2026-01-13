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

enum class DeepseekMoeGateEltwiseBinaryMode {
    COPY,
    RELOAD,
};

// local function declarations
template <EltwiseBinaryType eltwise_binary_type>
inline void deepseek_moe_gate_eltwise_binary_configure_addrmod() {
    // Use srcA for data movement
    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        addr_mod_t{
            .srca = {.incr = 8},
            .srcb = {.incr = 8},
            .dest = {.incr = 8},
        }
            .set(ADDR_MOD_0);
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_1);
    }
}

template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void deepseek_moe_gate_eltwise_binary_reuse_dest_as_src() {
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA) {
        TTI_STALLWAIT(
            p_stall::STALL_MATH,
            p_stall::WAIT_SFPU | p_stall::SRCA_VLD);  // MOVD2A for a whole face assumes unpacker will set a dummy
                                                      // data_valid, so we want to wait on that
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 0, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 0);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 4, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 4);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 8, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 8);
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 12, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 12);
    } else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB) {
        TTI_STALLWAIT(
            p_stall::STALL_MATH,
            p_stall::WAIT_SFPU | p_stall::SRCB_VLD);  // MOVD2B for a whole face assumes unpacker will set a dummy
                                                      // data_valid, so we want to wait on that
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
    }
}

template <EltwiseBinaryType eltwise_binary_type, DstSync Dst, bool is_fp32_dest_acc_en, int NUM_FIDELITY_PHASES = 0>
inline void _llk_math_deepseek_moe_gate_eltwise_binary_(
    const std::uint32_t num_faces, uint dst_index, const bool clear_fp32_dst_acc) {
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);
    constexpr uint32_t ZERO_ACC_MODE = p_zeroacc::CLR_16;
    static_assert(!(eltwise_binary_type == ELWMUL && high_fidelity), "High fidelity is not supported for ELWMUL");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        ckernel_template::run();
    }
    math::clear_dst_reg_addr();
}

template <EltwiseBinaryType eltwise_binary_type, DeepseekMoeGateEltwiseBinaryMode mode, int NUM_FIDELITY_PHASES = 0>
inline void deepseek_moe_gate_eltwise_binary_configure_mop(
    const std::uint32_t acc_to_dest = 0, const std::uint32_t num_faces = 4) {
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr bool high_fidelity = (NUM_FIDELITY_PHASES > 0);
    const uint addr_mod = ADDR_MOD_0;
    constexpr uint innerloop = 16 >> 3;  // 8 rows per eltwise op at a time.
    uint outerloop = num_faces;
    auto broadcast_type = p_elwise::SRCB_NO_BCAST;

    constexpr uint32_t dst_tile_offset = 64;  // 1 tile x 64 rows per tile
    constexpr uint32_t dst_math_offset = 2 * dst_tile_offset;

    uint math_op;

    // TODO: Probably not worth it to use a replay buffer/mop for this
    // Just hardcode the math ops in _llk_math_deepseek_moe_gate_eltwise_binary_ since we only have 1 face
    if constexpr (eltwise_binary_type == ELWADD) {
        math_op = TT_OP_ELWADD(0, acc_to_dest, broadcast_type, addr_mod, dst_math_offset);

    } else if constexpr (eltwise_binary_type == ELWSUB) {
        math_op = TT_OP_ELWSUB(0, acc_to_dest, broadcast_type, addr_mod, dst_math_offset);
    } else if constexpr (eltwise_binary_type == ELWMUL) {
        static_assert(!high_fidelity, "High fidelity is not supported for ELWMUL");
        math_op = TT_OP_ELWMUL(0, 0, broadcast_type, addr_mod, dst_math_offset);
    }
    if constexpr (mode == DeepseekMoeGateEltwiseBinaryMode::RELOAD) {
        load_replay_buf(
            ckernel::math::replay_buf_offset,  // replay buffer offset
            5,
            [] { deepseek_moe_gate_eltwise_binary_reuse_dest_as_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(); });
        uint replay_instr = lltt::replay_insn(math::replay_buf_offset, 5);
        ckernel_template tmp(outerloop, innerloop, math_op);
        tmp.set_start_op(replay_instr);
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    } else {
        ckernel_template tmp(outerloop, innerloop, TT_OP_MOVA2D(0, 0, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0), math_op);
        tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
}

template <EltwiseBinaryType eltwise_binary_type, DeepseekMoeGateEltwiseBinaryMode mode, int MATH_FIDELITY_DESC = 0>
inline void _llk_math_deepseek_moe_gate_eltwise_binary_init_(
    const std::uint32_t num_faces, const std::uint32_t acc_to_dest) {
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);

    deepseek_moe_gate_eltwise_binary_configure_addrmod<eltwise_binary_type>();

    if constexpr (
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL)) {
        deepseek_moe_gate_eltwise_binary_configure_mop<eltwise_binary_type, mode, MATH_FIDELITY_PHASES>(
            acc_to_dest, num_faces);
    }

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}
