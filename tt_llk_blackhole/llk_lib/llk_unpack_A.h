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
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_mop_config_(
    const bool transpose_of_faces, const std::uint32_t num_faces, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format = 0)
{
    static_assert(
        !((BType != BroadcastType::NONE) && acc_to_dest && (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)), "Not supported configuration!");
    static_assert(
        !(((acc_to_dest) || (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE)) && (unpack_to_dest)),
        "Not supported configuration when unpacking to dest!");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    static constexpr std::uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_to_dest =
        TT_OP_UNPACR(SrcA, 0b00010001 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // ch0/ch1 z_inc
    static constexpr std::uint32_t unpack_srca_to_dest_column =
        TT_OP_UNPACR(SrcA, 0b00100010 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // ch0/ch1 z_inc
    static constexpr std::uint32_t unpack_srca_to_dest_transpose_of_faces =
        TT_OP_UNPACR(SrcA, 0b00010010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch1_z+=1, ch0_z+=2
    static constexpr std::uint32_t unpack_srca_set_dvalid = TT_OP_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t unpack_srcb =
        TT_OP_UNPACR(SrcB, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_inc_z_0 =
        TT_OP_UNPACR(SrcB, 0b0 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    static constexpr std::uint32_t srca_set_z_1           = TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001); // set srcA ch0_z = 1
    static constexpr std::uint32_t srcb_set_z_2           = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001); // set srcB ch0_z = 2
    static constexpr std::uint32_t srcb_clear_z           = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001); // set srcB ch0_z = 0

    if (unpack_to_dest && is_32bit_input(unpack_src_format, unpack_dst_format))
    {
        if (transpose_of_faces && num_faces == 4)
        {
            const std::uint32_t outerloop = 2;
            const std::uint32_t innerloop = 2;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest_transpose_of_faces);
            tmp.set_end_op(TT_OP_SETADCZW(p_setadc::UNP_A, 0, 2, 0, 1, 0b0101));
            tmp.program();
        }
        else if (BType == BroadcastType::ROW || BType == BroadcastType::SCALAR)
        {
            constexpr std::uint32_t outerloop = BType == BroadcastType::ROW ? 2 : 1;
            constexpr std::uint32_t innerloop = 1;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest);
            tmp.program();
        }
        else if (BType == BroadcastType::COL)
        {
            constexpr std::uint32_t outerloop = 2;
            constexpr std::uint32_t innerloop = 1;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest_column);
            tmp.program();
        }
        else
        {
            const std::uint32_t outerloop     = num_faces;
            constexpr std::uint32_t innerloop = 1;
            ckernel_template tmp(outerloop, innerloop, unpack_srca_to_dest);
            tmp.program();
        }
    }
    else if constexpr (BType == BroadcastType::COL)
    {
        if constexpr (acc_to_dest)
        {
            constexpr std::uint32_t innerloop = 1;
            constexpr std::uint32_t outerloop = 2; // TODO: add support for num_faces, add support for dest to srcB
            ckernel_template tmp(outerloop, innerloop, unpack_srca_set_dvalid, unpack_srca_set_dvalid);
            tmp.set_start_op(unpack_srcb);
            tmp.set_end_op(srcb_set_z_2);
            tmp.program();
        }
        else
        {
            constexpr std::uint32_t innerloop = 1;
            constexpr std::uint32_t outerloop = 1; // TODO: add support for num_faces
            ckernel_template tmp(outerloop, innerloop, unpack_srcb, srcb_set_z_2);
            tmp.set_start_op(unpack_srca_set_dvalid);
            tmp.set_end_op(unpack_srcb);
            tmp.program();
        }
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        constexpr std::uint32_t innerloop = 2;
        constexpr std::uint32_t outerloop = 2; // TODO: add support for num_faces
        if constexpr (acc_to_dest)
        {
            ckernel_template tmp(outerloop, innerloop, unpack_srcb, unpack_srca_set_dvalid);
            tmp.set_end_op(srcb_clear_z);
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, unpack_srcb);
            tmp.set_end_op(srcb_clear_z);
            tmp.program();
        }
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        static_assert((!acc_to_dest) && "accumulate into dest with broadcast scaler is not supported!");
        constexpr std::uint32_t outerloop = 1;
        constexpr std::uint32_t innerloop = 1;
        ckernel_template tmp(outerloop, innerloop, unpack_srcb_inc_z_0);
        tmp.program();
    }
    else
    {
        if (transpose_of_faces)
        {
            constexpr std::uint32_t replay_buf_len = 2;
            load_replay_buf(
                0,
                replay_buf_len,
                [num_faces]
                {
                    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                    if (num_faces > 2)
                    {
                        TTI_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch0_z+=2
                    }
                    else
                    {
                        TTI_UNPACR(SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc srcA ch0_z+=1
                    }
                });

            const std::uint32_t outerloop = num_faces < 4 ? 1 : 2;
            const std::uint32_t innerloop = num_faces < 2 ? 1 : 2;
            ckernel_template tmp(outerloop, innerloop, lltt::replay_insn(0, replay_buf_len)); // Unpack faces 0/2 && 1/3 to srcA
                                                                                              // or 0/1 for 2 face tile
            if (num_faces > 2)
            {
                tmp.set_end_op(srca_set_z_1);
            }
            tmp.program();
        }
        else
        {
            if constexpr (acc_to_dest)
            {
                static constexpr std::uint32_t unpack_srca_reuse =
                    (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA) ? unpack_srca_set_dvalid : unpack_srca;

                static constexpr std::uint32_t unpack_srcb_reuse =
                    (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB) ? unpack_srcb_set_dvalid : unpack_srcb;

                const std::uint32_t outerloop     = num_faces;
                constexpr std::uint32_t innerloop = 1;
                ckernel_template tmp(outerloop, innerloop, unpack_srca_reuse, unpack_srcb_reuse);
                tmp.program();
            }
            else
            {
                const std::uint32_t outerloop     = num_faces;
                constexpr std::uint32_t innerloop = 1;
                ckernel_template tmp(outerloop, innerloop, unpack_srcb_set_dvalid);
                tmp.set_start_op(unpack_srca);
                tmp.program();
            }
        }
    }
}

template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_init_(
    const std::uint32_t transpose_of_faces          = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    const std::uint32_t num_faces                   = 4,
    const std::uint32_t unpack_src_format           = 0,
    const std::uint32_t unpack_dst_format           = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    // Set transpose register to prevent state pollution
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose);

    // TODO NC: Find out why we need to disable src zero flags for uint16 dst format #960
    // bool disable_src_zero_flag_val = disable_src_zero_flag || (static_cast<uint>(unpack_dst_format) == static_cast<uint>(DataFormat::UInt16));
    // cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(disable_src_zero_flag_val ? 1 : 0);

    constexpr std::uint32_t UNP_SEL = (BType == BroadcastType::NONE || unpack_to_dest) ? p_setadc::UNP_A : p_setadc::UNP_B;
    if constexpr ((BType == BroadcastType::ROW || BType == BroadcastType::SCALAR) && unpack_to_dest) // ROW and SCALAR bcast will only unpack a single row
    {
        config_unpacker_x_end<UNP_SEL>(1);
    }
    else // base case is to upk the entire face
    {
        config_unpacker_x_end<UNP_SEL>(face_r_dim);
    }

    // TODO NC: Move to TRISC1 tt-metal#36411
    if constexpr (BType != BroadcastType::NONE && unpack_to_dest)
    {
        _llk_unpack_dbg_feature_disable_();
    }
    _llk_unpack_A_mop_config_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(transpose_of_faces > 0, num_faces, unpack_src_format, unpack_dst_format);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_A_uninit_(const std::uint32_t face_r_dim)
{
    // Unpack A is used for all single unpacker operations, except bcast, since bcast HW feature is only available on unpacker B
    constexpr std::uint32_t UNP_SEL = (BType == BroadcastType::NONE) ? p_setadc::UNP_A : p_setadc::UNP_B;
    // TODO NC: Issue tt-llk#1036 will make this transient
    TT_SETADCXX(UNP_SEL, face_r_dim * FACE_C_DIM - 1, 0x0);
}

template <
    BroadcastType BType                          = BroadcastType::NONE,
    bool acc_to_dest                             = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest                          = false>
inline void _llk_unpack_A_(const std::uint32_t address, const std::uint32_t unpack_src_format = 0, const std::uint32_t unpack_dst_format = 0)
{
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Set upk0/1 L1 read addr
    if constexpr (((BType == BroadcastType::NONE) && (!acc_to_dest)) || binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB || unpack_to_dest)
    {
        const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
        cfg[upk0_reg]                = address;
    }
    else
    {
        const std::uint32_t upk1_reg = (unp_cfg_context == 0) ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32;
        cfg[upk1_reg]                = address;
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    if constexpr (unpack_to_dest)
    {
        if (is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            set_dst_write_addr(unp_cfg_context, unpack_dst_format);
            wait_for_dest_available();
        }
    }

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    ckernel::ckernel_template::run();

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    if (unpack_to_dest)
    {
        if (is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            unpack_to_dest_tile_done(unp_cfg_context);
        }
    }

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
