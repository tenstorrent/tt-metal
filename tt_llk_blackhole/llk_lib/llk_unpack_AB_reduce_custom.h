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
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * Configures MOP (Macro Operation) for block-based reduce_max_row unpacking operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce unpacking LLK MOP configuration.
 * Use the standard _llk_unpack_AB_mop_config_ for general-purpose block reduction operations.
 */
template <uint32_t block_ct_dim>
inline void _llk_unpack_AB_reduce_block_max_row_mop_config_()
{
    // Constraint on the outerloop and innerloop dim
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    // Single UNPACR because TTI_SETADCXX for UNP_A is 1023, increment Z counter to point to the next tile, set dvalid each time
    static constexpr uint unpack_srca_op =
        TT_OP_UNPACR(SrcA, 0b00000001 /* Z_ch0_inc and Z_ch1_inc */, 0, 0, 0, 1, 1 /* Set Dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    constexpr uint32_t outerloop = block_ct_dim;
    const uint32_t innerloop     = 1; // Unpack tile by tile of the input operand into SrcA
    ckernel_template tmp(outerloop, innerloop, unpack_srca_op);
    tmp.program();
}

/**
 * Initializes unpacker configuration for block-based reduce_max_row operations.
 * Sets up tile dimensions and saves unpacker state that will be modified during operation.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce unpacking LLK initialization.
 * Use the standard _llk_unpack_AB_reduce_init_ for general-purpose reduction operations.
 */
template <uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_AB_reduce_block_max_row_init_()
{
    if constexpr (is_fp32_dest_acc_en)
    {
        // Set necessary config regs for MOVB2D hi16/lo16 to work
        _llk_unpack_dbg_feature_disable_();
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    }
    // REDUCE_ROW requires transpose itself; additionally, within_face_16x16_transpose flag could require transpose;
    // if we have the flag set with REDUCE_ROW, we don't need to do anything
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1);

    TTI_SETADCXX(p_setadc::UNP_B, FACE_R_DIM * FACE_C_DIM - 1, 0x0);       // Unpack a single face of a scaler
    TTI_SETADCXX(p_setadc::UNP_A, 4 * (FACE_R_DIM * FACE_C_DIM) - 1, 0x0); // Unpack a whole tile of an operand

    // save the following state that is going to be modified:
    // tile x, y, and z dims for both unpackers
    // CH1 Z stride for both unpackers
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);

    TTI_SETDMAREG(0, 1 * FACE_C_DIM * FACE_R_DIM, 0, LO_16(p_gpr_unpack::TMP0));
    TTI_SETDMAREG(0, 1 * FACE_C_DIM * FACE_R_DIM, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_SETDMAREG(0, 4 /* y_dim */, 0, LO_16(p_gpr_unpack::TMP0));
    TTI_SETDMAREG(0, 1 /* z_dim */, 0, HI_16(p_gpr_unpack::TMP0));
    TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);

    _llk_unpack_AB_reduce_block_max_row_mop_config_<block_ct_dim>(); // Unpack operand and scaler
}

/**
 * Performs unpacking for block-based reduce_max_row operation across multiple tiles.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_unpack_AB_ LLK.
 * Use the standard _llk_unpack_AB_ in a loop for general-purpose block reduction operations.
 */
inline void _llk_unpack_AB_reduce_block_max_row_(const std::uint32_t address_a, const std::uint32_t address_b)
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

    TTI_UNPACR(SrcB, 0b00000000 /* Z_ch0_inc and Z_ch1_inc */, 0, 0, 0, 1, 1 /* Set Dvalid */, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    // Run MOP
    ckernel::ckernel_template::run();

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}

/**
 * Uninitializes block-based reduce_max_row unpacker operation.
 * Restores the unpacker state that was saved during initialization.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce unpacking cleanup.
 * Standard _llk_unpack_AB_reduce_init_ operations typically don't require explicit cleanup.
 */
inline void _llk_unpack_AB_reduce_block_max_row_uninit_()
{
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32);
    TTI_WRCFG(p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1);
    TTI_SETADCXX(p_setadc::UNP_AB, (FACE_R_DIM * FACE_C_DIM) - 1, 0x0);
}
