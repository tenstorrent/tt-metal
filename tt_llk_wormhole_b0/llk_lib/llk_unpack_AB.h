// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../../common/tensor_shape.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "lltt.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Configure the MOP (Micro-Operation Program) for unpacking two source operands A and B
 *
 * Sets up the unpacker MOP to handle various broadcast modes and transpose configurations.
 * The MOP programs the sequence of unpack operations based on tile geometry and broadcast type.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param transpose_of_faces: Whether to transpose faces (reorder faces 0,2,1,3)
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape &tensor_shape)
{
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    validate_tensor_shape_tile_dependent_ops_(tensor_shape);

    if (transpose_of_faces)
    {
        LLK_ASSERT(num_faces_r_dim == num_faces_c_dim, "num_faces_r_dim must be equal to num_faces_c_dim when transpose_of_faces is true");
        LLK_ASSERT(
            num_faces_c_dim == 2,
            "num_faces_c_dim has to be 2 with transpose due to stride limitations in UNPACR instruction, this limitation can be removed when TensiorShapes are "
            "passed compile time");
    }

    static constexpr std::uint32_t unpack_srca = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srcb = TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr std::uint32_t unpack_srca_transpose =
        TT_OP_UNPACR(SrcA, 0b10 /*This is an inc of 2, which is meant to be num_faces_c_dim*/, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    const std::uint32_t outerloop = transpose_of_faces ? num_faces_c_dim : num_faces_r_dim;
    const std::uint32_t innerloop = transpose_of_faces ? num_faces_r_dim : num_faces_c_dim;
    const std::uint32_t srca_op   = transpose_of_faces ? unpack_srca_transpose : unpack_srca;

    // Helper to set end op(s) based on transpose mode
    auto set_end_op_with_transpose = [&](ckernel_template &tmp, std::uint32_t primary_end_op)
    {
        if (transpose_of_faces)
        {
            tmp.set_end_ops(primary_end_op, TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001));
        }
        else
        {
            tmp.set_end_op(primary_end_op);
        }
    };

    if constexpr (BType == BroadcastType::COL)
    {
        // COL broadcast: First col in Src B face is broadcast across A faces in the same row
        LLK_ASSERT(
            num_faces_c_dim >= num_faces_r_dim,
            "If num_faces_c_dim is less than num_faces_r_dim (i.e 32x16), then BROADCAST_TYPE::COL is not supported, Can be fixed in the future");
        ckernel_template tmp(outerloop, innerloop, srca_op);
        tmp.set_start_op(unpack_srcb);

        if (num_faces_c_dim < MAX_NUM_FACES_C_DIM)
        {
            set_end_op_with_transpose(
                tmp, TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 1 /*this should be num_faces_c_dim, but can't pass it here until its compile time*/, 0b0001));
        }
        else
        {
            set_end_op_with_transpose(
                tmp, TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2 /*this should be num_faces_c_dim, but can't pass it here until its compile time*/, 0b0001));
        }

        tmp.program();
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        LLK_ASSERT(
            num_faces_c_dim >= num_faces_r_dim,
            "If num_faces_c_dim is less than num_faces_r_dim (i.e 32x16), then BROADCAST_TYPE::ROW is not supported, Can be fixed in the future");
        // ROW broadcast: First row in Src B face is broadcast across A faces in the same column
        static constexpr std::uint32_t unpack_srcb_clear_z = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001);

        ckernel_template tmp(outerloop, innerloop, unpack_srcb, srca_op);
        set_end_op_with_transpose(tmp, unpack_srcb_clear_z);
        tmp.program();
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        // SCALAR broadcast: single B value broadcast to all A faces
        LLK_ASSERT(!transpose_of_faces, "SrcA transpose is not supported with scalar broadcast");

        ckernel_template tmp(1, tensor_shape.total_num_faces(), unpack_srca);
        tmp.set_start_op(unpack_srcb);
        tmp.program();
    }
    else // BType == BroadcastType::NONE
    {
        // NONE: no broadcast, A and B faces are paired 1:1
        if (transpose_of_faces)
        {
            static constexpr std::uint32_t srca_set_z = TT_OP_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 1, 0b0001);
            // Flip r & c dimension due to transpose of SrcA, SrcA unpack increments L1 pointer by num_faces_c_dim
            ckernel_template tmp(num_faces_c_dim, num_faces_r_dim, unpack_srca_transpose, unpack_srcb);
            tmp.set_end_op(srca_set_z);
            tmp.program();
        }
        else
        {
            ckernel_template tmp(num_faces_r_dim, num_faces_c_dim, unpack_srca, unpack_srcb);
            tmp.program();
        }
    }
}

/**
 * @brief Initialize unpacker to unpack two source operands A and B into SrcA and SrcB registers
 *
 * Configures the unpacker hardware for dual-operand unpacking with support for various
 * broadcast modes and optional transpose. Sets up number of datums to unpack based on face dimensions.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @param transpose: Whether to transpose within each face (0 = no transpose, >0 = transpose)
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t transpose = 0)
{
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    validate_tensor_shape_tile_dependent_ops_(tensor_shape);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose); // transpose within the face

    config_unpacker_x_end<p_setadc::UNP_AB>(tensor_shape.face_r_dim);

    _llk_unpack_AB_mop_config_<BType>(transpose > 0, tensor_shape); // transpose of faces 0,2,1,3
}

/**
 * @brief Uninitialize unpacker after AB unpacking operations
 *
 * Resets the unpacker address counters for both SrcA and SrcB to their default
 * tile element counts based on the provided tensor shapes.
 *
 * @param unpA_tensor_shape: Tensor shape for source A operand
 * @param unpB_tensor_shape: Tensor shape for source B operand
 */
inline void _llk_unpack_AB_uninit_(const ckernel::TensorShape unpA_tensor_shape, const ckernel::TensorShape unpB_tensor_shape)
{
    // TODO NC: Issue tt-llk#1036 will make this transient
    TT_SETADCXX(p_setadc::UNP_A, unpA_tensor_shape.face_r_dim * unpA_tensor_shape.face_c_dim - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_tensor_shape.face_r_dim * unpB_tensor_shape.face_c_dim - 1, 0x0);
}

/**
 * @brief Unpack two tiles from L1 memory into SrcA and SrcB registers
 *
 * Performs the actual unpacking operation by programming base addresses and running
 * the configured MOP. Handles context switching and synchronization with the unpacker.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param address_a: L1 memory address of source A tile
 * @param address_b: L1 memory address of source B tile
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(const std::uint32_t address_a, const std::uint32_t address_b)
{
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111); // reset counters

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Validate and configure addresses
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);

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

inline void _llk_unpack_bcastA_B_mop_config_()
{
    // Setup address modifiers for unpacker instructions
    constexpr std::uint8_t ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0 = 0b00'00'00'00;
    constexpr std::uint8_t ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0 = 0b01'00'00'00; // Increment CH1_Y by 1 Y_STRIDE

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
    constexpr std::uint8_t ADDRMOD_CH1Y_1_CH1Z_0_CH0Y_0_CH0Z_0 = 0b01'00'00'00; // Increment CH1_Y by 1 Y_STRIDE

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

inline void _llk_unpack_bcastA_B_uninit_(const std::uint32_t y_stride = FACE_R_DIM * 2, const std::uint32_t face_r_dim = FACE_R_DIM)
{
    // Revisit default stride value in tt-llk#1015
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(y_stride);
    TT_SETADCXX(p_setadc::UNP_AB, face_r_dim * FACE_C_DIM - 1, 0x0);
}

inline void _llk_unpack_bcastA_B_(const std::uint32_t address_a, const std::uint32_t address_b, std::uint32_t srca_reuse_count = 4)
{
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW)); // reset counters
    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y));  // Clear Y counter on src side

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Validate and configure addresses
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);

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

    std::uint32_t unpack_mask = 0xFFFE;

    ckernel_unpack_template::run(srca_reuse_count, unpack_mask);

    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y)); // Clear all counters

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
