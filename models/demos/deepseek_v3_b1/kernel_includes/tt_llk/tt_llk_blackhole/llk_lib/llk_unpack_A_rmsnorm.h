// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <
    uint32_t num_tiles,
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void _llk_unpack_A_rmsnorm_mop_config_(
    const bool transpose_of_faces,
    const std::uint32_t num_faces,
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format = 0) {
    static_assert(
        ((BType == BroadcastType::SCALAR) && acc_to_dest &&
         (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)),
        "Not supported configuration!");
    static_assert(
        (((BType == BroadcastType::NONE) && (!acc_to_dest) &&
          (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)) ||
         (!unpack_to_dest)),
        "Not supported configuration when unpacking to dest!");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    static constexpr uint unpack_srca = TT_OP_UNPACR(
        SrcA,
        0b1 /*Z inc*/,
        0,
        0,
        0,
        1 /* Set OvrdThreadId*/,
        1 /*Set Dvalid*/,
        p_unpacr::RAREFYB_DISABLE,
        0,
        0,
        0,
        0,
        1);
    static constexpr uint unpack_srca_to_dest = TT_OP_UNPACR(
        SrcA,
        0b00010001 /*Z inc*/,
        0,
        0,
        0,
        1 /* Set OvrdThreadId*/,
        0 /*Set Dvalid*/,
        p_unpacr::RAREFYB_DISABLE,
        0,
        0,
        0,
        0,
        1);  // ch0/ch1 z_inc
    static constexpr uint unpack_srca_to_dest_transpose_of_faces = TT_OP_UNPACR(
        SrcA, 0b00010010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);  // inc srcA ch1_z+=1, ch0_z+=2
    static constexpr uint unpack_srca_set_dvalid =
        TT_OP_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr uint unpack_srcb = TT_OP_UNPACR(
        SrcB,
        0b1 /*Z inc*/,
        0,
        0,
        0,
        1 /* Set OvrdThreadId*/,
        1 /*Set Dvalid*/,
        p_unpacr::RAREFYB_DISABLE,
        0,
        0,
        0,
        0,
        1);
    static constexpr uint unpack_srcb_inc_z_0 = TT_OP_UNPACR(
        SrcB,
        0b0 /*Z inc*/,
        0,
        0,
        0,
        1 /* Set OvrdThreadId*/,
        1 /*Set Dvalid*/,
        p_unpacr::RAREFYB_DISABLE,
        0,
        0,
        0,
        0,
        1);
    static constexpr uint unpack_srcb_set_dvalid =
        TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr uint srca_set_z_1 = TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001);  // set srcA ch0_z = 1
    static constexpr uint srcb_set_z_2 = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001);  // set srcB ch0_z = 2
    static constexpr uint srcb_clear_z = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001);  // set srcB ch0_z = 0
    if (num_faces == 1) {
        constexpr uint32_t outerloop = 1;
        constexpr uint32_t innerloop = num_tiles;
        ckernel_template tmp(outerloop, innerloop, unpack_srca);
        tmp.set_start_op(unpack_srcb_set_dvalid);
        tmp.program();
    } else {
        constexpr uint32_t outerloop = 1;
        const uint32_t innerloop = num_tiles * num_faces / 2;
        ckernel_template tmp(outerloop, innerloop, unpack_srca, unpack_srca);
        tmp.set_start_op(unpack_srcb_set_dvalid);
        tmp.program();
    }
}

template <
    uint32_t num_tiles,
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void _llk_unpack_A_rmsnorm_init_(
    const std::uint32_t transpose_of_faces = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces = 4,
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0) {
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    // Set transpose register to prevent state pollution
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose);

    // TODO NC: Find out why we need to disable src zero flags for uint16 dst format #960
    // bool disable_src_zero_flag_val = disable_src_zero_flag || (static_cast<uint>(unpack_dst_format) ==
    // static_cast<uint>(DataFormat::UInt16));
    // cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(disable_src_zero_flag_val ? 1 : 0);

    constexpr std::uint32_t UNP_SEL = (BType == BroadcastType::NONE) ? p_setadc::UNP_A : p_setadc::UNP_B;
    config_unpacker_x_end<UNP_SEL>(face_r_dim);

    _llk_unpack_A_rmsnorm_mop_config_<num_tiles, BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces > 0, num_faces, unpack_src_format, unpack_dst_format);
}
