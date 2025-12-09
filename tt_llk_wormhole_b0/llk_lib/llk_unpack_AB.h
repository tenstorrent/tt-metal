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
#include "lltt.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces = false, const std::uint32_t num_faces = 4, const bool narrow_tile = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    static constexpr uint unpack_srca = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srcb = TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    if constexpr (BType == BroadcastType::COL)
    {
        static constexpr uint unpack_srcb_set_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 2, 0b0001);
        const uint32_t outerloop                = num_faces < 4 ? 1 : 2;
        const uint32_t innerloop                = num_faces < 2 ? 1 : 2;
        ckernel_template tmp(outerloop, innerloop, unpack_srca);
        tmp.set_start_op(unpack_srcb);
        if (narrow_tile)
        {
            tmp.set_end_op(unpack_srcb); // Read face 1
        }
        else
        {
            tmp.set_end_op(unpack_srcb_set_z);
        }
        tmp.program();
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        static constexpr uint unpack_srcb_clear_z  = TT_OP_SETADCZW(0b010, 0, 0, 0, 0, 0b0001);
        static constexpr uint unpack_srcb_no_z_inc = TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        const uint32_t outerloop                   = num_faces < 4 ? 1 : 2;
        const uint32_t innerloop                   = num_faces < 2 ? 1 : 2;
        ckernel_template tmp(outerloop, innerloop, narrow_tile ? unpack_srcb_no_z_inc : unpack_srcb, unpack_srca);
        tmp.set_end_op(unpack_srcb_clear_z);
        tmp.program();
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        const uint32_t outerloop = 1;
        const uint32_t innerloop = num_faces;
        ckernel_template tmp(outerloop, innerloop, unpack_srca);
        tmp.set_start_op(unpack_srcb);
        tmp.program();
    }
    else
    {
        if (transpose_of_faces)
        {
            static constexpr uint srca_set_z         = TT_OP_SETADCZW(0b001, 0, 0, 0, 1, 0b0001);                                         // set z to 1
            static constexpr uint unpack_srca_skip_z = TT_OP_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // inc z by 2
            const uint32_t outerloop                 = num_faces < 4 ? 1 : 2;
            const uint32_t innerloop                 = num_faces < 2 ? 1 : 2;
            ckernel_template tmp(outerloop, innerloop, num_faces < 4 ? unpack_srca : unpack_srca_skip_z, unpack_srcb);
            tmp.set_end_op(srca_set_z);
            tmp.program();
        }
        else
        {
            constexpr uint32_t outerloop = 1;
            const uint32_t innerloop     = num_faces;
            ckernel_template tmp(outerloop, innerloop, unpack_srca, unpack_srcb);
            tmp.program();
        }
    }
}

template <bool is_fp32_dest_acc_en, StochRndType stoch_rnd_mode = StochRndType::None>
inline void _llk_unpack_AB_hw_configure_(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t num_faces                   = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr bool is_row_pool  = false;
    constexpr bool stoch_rnd_en = (stoch_rnd_mode == StochRndType::All);
    constexpr bool fpu_srnd_en  = stoch_rnd_en || (stoch_rnd_mode == StochRndType::Fpu);
    constexpr bool pack_srnd_en = stoch_rnd_en || (stoch_rnd_mode == StochRndType::Pack);
    configure_unpack_AB<is_fp32_dest_acc_en, is_row_pool, fpu_srnd_en, pack_srnd_en>(
        unpA_src_format, unpB_src_format, unpA_dst_format, unpB_dst_format, face_r_dim, face_r_dim, within_face_16x16_transpose, num_faces, num_faces);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(
    const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4, const bool narrow_tile = false, const std::uint32_t transpose = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose); // transpose within the face

    constexpr std::uint32_t UNP_SEL = p_setadc::UNP_AB;
    config_unpacker_x_end<UNP_SEL>(face_r_dim);

    _llk_unpack_AB_mop_config_<BType>(transpose > 0, num_faces, narrow_tile); // transpose of faces 0,2,1,3
}

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(const std::uint32_t address_a, const std::uint32_t address_b)
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111); // reset counters

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Get tile address
    if (0 == unp_cfg_context)
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    ckernel::ckernel_template::run();

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}

/*************************************************************************
 * LLK sub_bcast_row_tile unpacker implementation for SDPA
 *************************************************************************/

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_bcastA_B_hw_config_(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t num_faces                   = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr bool is_row_pool = false;
    configure_unpack_AB<is_fp32_dest_acc_en, is_row_pool>(
        unpA_src_format, unpB_src_format, unpA_dst_format, unpB_dst_format, face_r_dim, face_r_dim, within_face_16x16_transpose, num_faces, num_faces);
}

inline void _llk_unpack_bcastA_B_mop_config_()
{
    // Setup address modifiers for unpacker instructions
    constexpr uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0 = 0b00'00'00'00;
    constexpr uint8_t ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0 = 0b01'00'00'00; // Increment CH1_Y by 1 Y_STRIDE

    /*

        Configuration of unpacker MOP. Setting halo and unpackB to true which
        enables usage of all unpack instrructions 0-3 and unpack_B instruction.

        Unpack_A instructions are set up as expected with reading F0R0 and F1R0 and setting dvalid.
        One unpack_A which is A3 is used for unpacking B in first iteration and then SKIP_A and B are
        used for other 3 unpacks on B.

    */

    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,                                                                                                                          // unpackB
        true,                                                                                                                          // unpackHalo
        TT_OP_REPLAY(0, 16, 0, 0),                                                                                                     // A0_instr
        TT_OP_NOP,                                                                                                                     // A1_instr
        TT_OP_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1), // A2_instr
        TT_OP_UNPACR(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1), // A3_instr // UNPACK_A3
        TT_OP_UNPACR(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1), // skipA_instr

        TT_OP_INCADCZW(p_setadc::UNP_B, 0, 0, 0, 4), // B_instr
        TT_OP_INCADCZW(p_setadc::UNP_B, 0, 0, 0, 4)  // skipB_instr
    );

    tmp.program();
}

inline void _llk_unpack_bcastA_B_init_()
{
    // Manual setup for unpacker A
    // Only srcA Y stride can be set because other strides are never used.
    // Stride is applied in each UNPACR instruction with ADDRMOD.
    // When applied it moves to next row on srcA side.
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(32);

    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM - 1, 0);              // Directly set unpacker A counter to unpack one row
    TTI_SETADCXX(p_setadc::UNP_B, TILE_R_DIM * TILE_C_DIM - 1, 0); // Directly set unpacker B counter to unpack whole tile

    // Setup address modifiers for unpacker instructions
    constexpr uint8_t ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0 = 0b01'00'00'00; // Increment CH1_Y by 1 Y_STRIDE

    /*

        Fill replay buffer with 8 unpack instructions but none of them sets dvalid.
        They are used for contiguous unpacking od faces.
        Last instruction that increments Y counter by Y_STRIDE on channel 0 is alose inside of replay.

        Intended use inside of MOP:

        TT_OP_REPLAY(0, 9, 0, 0) -> Unpack F0R0 row 8 times and move to F1
        TT_OP_REPLAY(0, 7, 0, 0) ->  Unpack F1R0 7 times
        TT_OP_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1), -> Unpack F1R0 once and set dvalid

        As can be seen before dvalid needs to be set replay is called with 7 iterations and 8th will be one with dvalid = 1.


    */

    lltt::record<lltt::NoExec>(0, 16);
    // ************************************
    // F0
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_INCADCXY(p_setadc::UNP_A, 0, 0, 1, 0); // Increment Y to point to next needed data in L1

    // F1
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 0 /* dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    // dvalid will be set by UNPACR instruction in MOP
    // ************************************

    _llk_unpack_bcastA_B_mop_config_();
}

inline void _llk_unpack_bcastA_B_(const std::uint32_t address_a, const std::uint32_t address_b, uint32_t srca_reuse_count = 4)
{
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW)); // reset counters
    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y));  // Clear Y counter on src side

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Get tile address
    if (0 == unp_cfg_context)
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    /*
        Run the MOP in following way: The second parameter of ckernel_unpack_template::run specifies the mask for the unpacking operations.
        In this case bit 1 is set to 1 which means that unacker MOP will execute "else" part of it's loop on second pass.

        LOOP:
        if (!zmask[iteration]):
            UNPACR_A0
            UNPACR_A1
            UNPACR_A2
            UNPACR_A3
            UNPACR_B
        else
            SKIP_A
            SKIP_B

        In iteration 0 zmask will be 0 so unopacker will execute first 5 instructions. Those will unpack F0R0 and F1R0 into srcA and set dvalid.
        After that it will unpack full tile on B on increment Z counter on B so it moves to next tile.
        Next iterations have zmask on 1 and execute SKIP instructions which are just unpacks on B and after every unpack increment of Z counter on CH0

        The full unrolled code is:

        TT_OP_REPLAY(0, 16, 0, 0),
        TT_OP_NOP,
        TT_OP_UNPACR(SrcA, ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1),

        First B tile

        TT_OP_UNPACR(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1),
        TT_OP_INCADCZW(p_setadc::UNP_B, 0, 0, 0, 4),

        Other B tiles

        TT_OP_UNPACR(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1),
        TT_OP_INCADCZW(p_setadc::UNP_B, 0, 0, 0, 4)
        TT_OP_UNPACR(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1),
        TT_OP_INCADCZW(p_setadc::UNP_B, 0, 0, 0, 4)
        TT_OP_UNPACR(SrcB, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1),
        TT_OP_INCADCZW(p_setadc::UNP_B, 0, 0, 0, 4)
        ...

    */

    uint32_t unpack_mask = 0xFFFE;

    ckernel_unpack_template::run(srca_reuse_count, unpack_mask);

    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y)); // Clear all counters

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
