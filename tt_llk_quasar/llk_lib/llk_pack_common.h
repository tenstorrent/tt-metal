// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cpack_common.h"
using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief Programs packer l1 info & math destination register format
 * @tparam PACK_SEL: Sets which packer to configure. values = p_pacr::PACK0/PACK1
 * @param tdma_desc: Contains L1 buffer descriptor information & destination register format
 */

template <uint32_t PACK_SEL>
inline void _llk_pack_hw_configure_(const tdma_descriptor_t& tdma_desc)
{
    static_assert((PACK_SEL == p_pacr::PACK0) || (PACK_SEL == p_pacr::PACK1), "PACK_SEL can only be set to p_pacr::PACK0/PACK1");

    // Turn on automatic Tensix-TRISC synchronization
    set_ttsync_enables<TRACK_ALL>(ckernel::pack::TRISC_ID);

    // Populate the buffer descriptor table
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    // RT: make defines to aggregate the packer input format address, to make the below a single function
    // Program math destination register format
    if constexpr (PACK_SEL == p_pacr::PACK0)
    {
        cfg_rmw(THCON_PACKER0_REG0_IN_DATA_FORMAT_RMW, static_cast<uint8_t>(tdma_desc.reg_data_format));
    }
    else
    {
        cfg_rmw(THCON_PACKER1_REG0_IN_DATA_FORMAT_RMW, static_cast<uint8_t>(tdma_desc.reg_data_format));
    }
}

/**
 * @brief Clears the data valid for destination register after Packer 0 is done packing
 * and zeroes out the dest bank(s) used by packer 0
 * @tparam DST: Destination register buffering mode, values = [DstSync::SyncHalf, DstSync::SyncFull]
 * @tparam IS_FP32_MATH_DEST_EN: flag to show if math destination register is set to float32 mode
 *
 * IMPORTANT NOTE:
 * 1) Uses ADDR_MOD_0 from math thread, but ZEROACC here only does CLR_HALF or CLR_ALL mode, addr_mod should not matter
 * It is the duty of the math operation to clear the counters set by addrmods before using the cleared bank
 * 2) Do not mix this function with the packer semaphore synchronization functions such as _llk_pack_dest_semaphore_section_done_
 * This function uses the dest data valid client synchronization scheme and updates bank ids accordingly, Only 1 type of sync scheme
 * can be used at a time: data valids or semaphores
 **/
template <DstSync DST, bool IS_FP32_MATH_DEST_EN>
inline void _llk_pack_dest_dvalid_section_done_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::WAIT_SFPU, p_stall::PACK);

    constexpr uint ZEROACC_CLR_MODE = (DST == DstSync::SyncHalf) ? p_zeroacc::CLR_HALF : p_zeroacc::CLR_ALL;
    const uint dest_id              = (DST == DstSync::SyncHalf) ? dest_bank_id : 0;
    TT_ZEROACC(ZEROACC_CLR_MODE, IS_FP32_MATH_DEST_EN, 0, ADDR_MOD_0, dest_id);
    TTI_CLEARDVALID(0, 0, 0, 0, p_cleardvalid::PACK, 0);

    if (DST == DstSync::SyncHalf)
    {
        _update_dest_bank_id_();
    }
}

/**
 * @brief Configure packer edge mask programming for packer 0 with reduce operations
 * @tparam REDUCE_DIM: The reduce op dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 **/
template <ReduceDim REDUCE_DIM>
inline void _llk_pack_reduce_mask_config_()
{
    // Wait for packer to finish to avoid breaking its current configuration
    TTI_STALLWAIT(p_stall::STALL_CFG, 0, 0, p_stall::PACK0);

    // This register specifies edge masking mode.
    //  0x0 -> mask to 0
    //  0x1 -> mask to -inf

    // TODO: (RT) Clean this up using pack edge struct to match addresses
    //  Make it unified
    if constexpr (REDUCE_DIM == ReduceDim::REDUCE_ROW)
    {
        // This register specifies which datums will not have the mask applied
        // The register is 16 bits, each bit corresponds to a datum in the 1x16 row in dest
        // 0xFFFE below means datum[0] preserves its values, datums[1:15] = 0
        cfg_rmw(THCON_PACKER0_REG1_EDGE_MASK1_RMW, 0xFFFE);

        // The registers below are 32 bits each, each 2 bits correspond to a row in a face
        // each 2 bits specify the mask that will be applied (there are 4 masks possible)
        // the registers below will have mask 01 applied to every row in the face
        cfg_rmw(THCON_PACKER0_REG2_EDGE_MASK_SELECT_FACE0_RMW, 0x55555555);
        cfg_rmw(THCON_PACKER0_REG2_EDGE_MASK_SELECT_FACE2_RMW, 0x55555555);
    }
    else if constexpr (REDUCE_DIM == ReduceDim::REDUCE_COL)
    {
        // The below mask mean all datums in a row preserve their value
        cfg_rmw(THCON_PACKER0_REG1_EDGE_MASK1_RMW, 0x0000);
        cfg_rmw(THCON_PACKER0_REG1_EDGE_MASK0_RMW, 0xFFFF);

        // For face 0 & face 1, only row 0 will have mask1 applied
        // Mask1 is configured to keep all datums in a row
        // rows[1-16] will have all of their datums masked to 0
        cfg_rmw(THCON_PACKER0_REG2_EDGE_MASK_SELECT_FACE0_RMW, 0x1);
        cfg_rmw(THCON_PACKER0_REG2_EDGE_MASK_SELECT_FACE1_RMW, 0x1);
    }
    else
    {
        // 0xFFFE below means datum[0] preserves its values, datums[1:15] = 0
        cfg_rmw(THCON_PACKER0_REG1_EDGE_MASK0_RMW, 0xFFFF);
        cfg_rmw(THCON_PACKER0_REG1_EDGE_MASK1_RMW, 0xFFFE);

        // For face 0, only row 0 will have mask1 applied
        // Mask1 is configured to only have datum[0] preserved
        // rows[1-16] will have all of their datums masked to 0
        cfg_rmw(THCON_PACKER0_REG2_EDGE_MASK_SELECT_FACE0_RMW, 0x1);
    }

    // Stall until all config instructions are done
    TTI_STALLWAIT(p_stall::PACK0, 0, 0, p_stall::TRISC_CFG);
}

/**
 * @brief Configure packer edge mask programming for packer 0 with reduce operations
 **/
inline void _llk_pack_reduce_mask_clear_()
{
    // Wait for packer to finish to avoid breaking its current configuration
    TTI_STALLWAIT(p_stall::STALL_CFG, 0, 0, p_stall::PACK0);

    // Edge mask mode is disabled
    // Mask0 is cleared to preserve values of all datums in a row
    cfg_rmw(THCON_PACKER0_REG1_EDGE_MASK0_RMW, 0x0000);

    // All packer faces are set to point to Mask0, which preserves all datums
    cfg_rmw(THCON_PACKER0_REG2_EDGE_MASK_SELECT_FACE0_RMW, 0x0);
    cfg_rmw(THCON_PACKER0_REG2_EDGE_MASK_SELECT_FACE1_RMW, 0x0);
    cfg_rmw(THCON_PACKER0_REG2_EDGE_MASK_SELECT_FACE2_RMW, 0x0);
    cfg_rmw(THCON_PACKER0_REG2_EDGE_MASK_SELECT_FACE3_RMW, 0x0);

    // Stall until all config instructions are done
    TTI_STALLWAIT(p_stall::PACK0, 0, 0, p_stall::TRISC_CFG);
}

/**
 * @brief: Configure packer to enable or disable l1 accumulation
 * @tparam PACK_SEL: Sets which packer to configure. values = p_pacr::PACK0/PACK1
 * @param l1_acc_en: if false -> l1 acc is disabled, true -> l1 acc enabled
 **/
template <uint32_t PACK_SEL>
inline void _llk_pack_set_l1_acc_(const bool l1_acc_en)
{
    if constexpr (PACK_SEL == p_pacr::PACK0)
    {
        cfg_rmw(THCON_PACKER0_REG0_L1_ACC_RMW, l1_acc_en);
    }
    else
    {
        cfg_rmw(THCON_PACKER1_REG0_L1_ACC_RMW, l1_acc_en);
    }
}

/**
 * All the following functions are added to enable Math <-> Pack synchronization
 * on destination register due to dest dvalid issue:
 * The following functions should be removed once the above issue is resolved
 */

// wait until math is done and has produced something to pack
inline void _llk_packer_wait_for_math_done_()
{
    TTI_SEMWAIT(p_stall::STALL_TDMA, p_stall::STALL_ON_ZERO, 0, semaphore::t6_sem(semaphore::MATH_PACK));
}

// Tell math that it can write again
template <uint WaitRes = p_stall::NOTHING>
inline void _llk_packer_set_math_semaphore_()
{
    t6_semaphore_get<WaitRes>(semaphore::MATH_PACK); // Indicate that packer is done and header is written into L1
}

/**
 * @brief Clear dest section after packer is done reading, signal to math dest section is ready to use
 * @tparam PACK_SEL: Sets which packer to configure. values = p_pacr::PACK0/PACK1
 * @tparam DST: Destination register buffering mode, values = [DstSync::SyncHalf, DstSync::SyncFull]
 * @tparam IS_FP32_MATH_DEST_EN: flag to show if math destination register is set to float32 mode
 */
template <uint32_t PACK_SEL, DstSync DST, bool IS_FP32_DEST_EN>
inline void _llk_pack_dest_semaphore_section_done_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::NOTHING, p_stall::PACK); // wait for pack to finish

    // TODO: (RT) Addrmod here is dangerous, can be overwritten by other pack operations
    //  Need to pick a addrmod, and assert no other math uses it
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    if constexpr (DST == DstSync::SyncFull)
    {
        TT_ZEROACC(p_zeroacc::CLR_ALL, IS_FP32_DEST_EN, 0, ADDR_MOD_7, 0);
    }
    else
    {
        static_assert(DST == DstSync::SyncHalf);
        TT_ZEROACC(p_zeroacc::CLR_HALF, IS_FP32_DEST_EN, 0, ADDR_MOD_7, (dest_bank_id) % 2);
    }

    // Tell math that it can write again
    _llk_packer_set_math_semaphore_();

    if constexpr (DST == DstSync::SyncHalf)
    {
        _update_dest_bank_id_();
        _set_packer_dest_registers_<PACK_SEL, DST>();
    }
}
