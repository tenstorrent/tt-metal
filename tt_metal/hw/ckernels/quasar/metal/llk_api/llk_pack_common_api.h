// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ckernel.h"
#include "llk_outputs.h"
#include "llk_pack_common.h"
#include "llk_dest_dvalid.h"
#include "llk_defs.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK PACK COMMON
 *************************************************************************/

/**
 * @brief Programs packer0 L1 information & math destination register format
 *
 * @tparam EN_32BIT_DEST: Set to true to use 32bit Destination register mode
 * @param pack_output The output DataFlow Buffer identifier
 */
template <bool EN_32BIT_DEST>
inline void llk_pack_hw_configure(const std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);

    // Program buffer descriptors for all 32 dataflow buffers, i is the logical dfb id.
    // Skip non-participating DFBs (gate matched the state in which A2 implicit-sync
    // passes; reverting to a plain unfiltered loop caused the implicit-sync 3-DFB
    // runtime to hang at credit-ack handshake). Loop bound is dfb::NUM_DFBS because
    // g_dfb_logical_to_compact[] is sized NUM_DFBS (=32) and NUM_CIRCULAR_BUFFERS
    // resolves to 64 on Quasar — GCC -Werror=aggressive-loop-optimizations rejects
    // the direct OOB array access at the gate.
    for (std::uint32_t i = 0; i < dfb::NUM_DFBS; ++i) {
        if (g_dfb_logical_to_compact[i] == 0xFF) {
            continue;
        }
        const DataFormat l1_data_format = static_cast<DataFormat>(pack_dst_format[i]);

        if (l1_data_format == DataFormat::Invalid) {
            continue;
        }

        // TODO: with multiple TCs are there multiple descriptors?
        // Same HW z_dim rule as the unpack side (see llk_unpack_hw_configure): 4 for a full 2x2 face
        // grid, 1 otherwise, so a tiny tile is one HW tile per face, which is the granularity
        // _llk_pack_ scales its L1 and dest tile indices to. Set using construct_tdma_desc helper.
        const ckernel::TensorShape tensor_shape = get_output_tensor_shape(i);
        const tdma_descriptor_t bd_td_val = ckernel::trisc::construct_tdma_desc(
            tensor_shape,
            get_local_dfb_interface(i).tc_slots[0].base_addr,
            static_cast<std::uint8_t>(l1_data_format),
            i,
            pack_src_format[i]);

        ckernel::trisc::_configure_buf_desc_table_(i, bd_td_val.buf_desc);
    }

    tdma_descriptor_t td_val;
    td_val.reg_data_format = static_cast<std::uint8_t>(pack_src_format[output_id]);
    _llk_pack_hw_configure_<p_pacr::PACK0, EN_32BIT_DEST>(td_val, ckernel::ReluConfig::none());
}

inline bool should_reconfig_pack_in_data_format(const std::uint32_t old_output, const std::uint32_t new_output) {
    const std::uint32_t old_output_id = get_output_id(old_output);
    const std::uint32_t new_output_id = get_output_id(new_output);
    return (pack_src_format[old_output_id] != pack_src_format[new_output_id]) ||
           (pack_dst_format[old_output_id] != pack_dst_format[new_output_id]);
}

/**
 * Reprograms packer THCON IN_DATA_FORMAT only (gasket); L1 format stays in buffer descriptors.
 */
template <[[maybe_unused]] bool EN_32BIT_DEST>
inline void llk_pack_reconfig_data_format(const std::uint32_t new_output) {
    const std::uint32_t output_id = get_output_id(new_output);
    _llk_pack_reconfig_data_format_<p_pacr::PACK0>(pack_src_format[output_id], pack_dst_format[output_id]);
}

template <bool EN_32BIT_DEST>
inline void llk_pack_reconfig_data_format(const std::uint32_t old_output, const std::uint32_t new_output) {
    if (!should_reconfig_pack_in_data_format(old_output, new_output)) {
        return;
    }
    llk_pack_reconfig_data_format<EN_32BIT_DEST>(new_output);
}

/**
 * @brief Clears the data valid for destination register after Packer 0 is done packing
 * and zeroes out the dest bank(s) used by packer 0
 *
 * @tparam DST: Destination register banking mode: SyncHalf = double banked (math/pack overlap), SyncFull = one bank
 *(serialized)
 * @tparam EN_32BIT_DEST: flag to show if Destination register is set to 32-bit mode
 *
 * @warning SYNC SCHEME: dest-dvalid. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 **/
template <DstSync DST, bool EN_32BIT_DEST>
inline void llk_pack_dest_dvalid_section_done() {
    _llk_pack_dest_dvalid_section_done_<DST, EN_32BIT_DEST>();
}

/**
 * @brief Waits until math has finished producing data for the current Destination Register section.
 * With dest-dvalid sync the hardware ring handles the wait via CLEARDVALID gating.
 */
inline void llk_packer_wait_for_math_done() {}

/**
 * @brief Signals that the packer has finished consuming the current Destination Register section.
 * Zeroes the consumed dest bank, then pulses the PACK dvalid client (waits for ring
 * condition, toggles) so the math client can proceed.  Uses a single pulse to avoid
 * the deadlock the double-pulse bank-reset form causes inside a configured ring.
 * @tparam EN_32BIT_DEST True if math destination is in 32-bit mode, false for 16-bit mode.
 */
template <bool EN_32BIT_DEST>
inline void llk_pack_dest_section_done() {
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::WAIT_SFPU, p_stall::PACK);
    constexpr std::uint32_t CLR_MODE = (DST_SYNC_MODE == DstSync::SyncHalf) ? p_zeroacc::CLR_HALF : p_zeroacc::CLR_ALL;
    if constexpr (DST_SYNC_MODE == DstSync::SyncFull) {
        TTI_ZEROACC(CLR_MODE, EN_32BIT_DEST, 0, ADDR_MOD_0, 0);
    } else {
        TT_ZEROACC(CLR_MODE, EN_32BIT_DEST, 0, ADDR_MOD_0, ckernel::pack::clear_dest_bank_id);
    }
    TTI_CLEARDVALID(0, 0, 0, 0, p_cleardvalid::PACK, 0);
    if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
        ckernel::pack::_update_clear_dest_bank_id_();
    }
}

/**
 * @brief Initializes the PACK side of the dest-dvalid ring and resets the dest section base.
 * Configures PACK as the non-start client of an {FPU, PACK} dvalid ring
 * with auto bank-ID toggle disabled (software bank management).
 */
inline void llk_pack_dest_init() {
    _llk_pack_dest_init_<p_pacr::PACK0, DST_SYNC_MODE>();
    _llk_dest_dvalid_exclude_<dest_dvalid_client::UNPACK>();
    _llk_dest_dvalid_exclude_<dest_dvalid_client::SFPU>();
    _llk_dest_dvalid_enable_<dest_dvalid_client::PACK>();
    cfg_rmw(PACK_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_RMW, 1);
    TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::NOTHING, p_stall::CFGEXU, p_stall::TRISC_CFG);
}

/**
 * @brief Configure packer ReLU at runtime from a packed uint32.
 * @param config Packed uint32: bits [1:0] = ReluType, bits [31:16] = threshold.
 */
TT_ALWAYS_INLINE void llk_pack_relu_config(const std::uint32_t config) {
    _llk_pack_relu_config_<p_pacr::PACK0, false /* EN_32B_DEST */>(ckernel::ReluConfig::from_packed(config));
}

TT_ALWAYS_INLINE void llk_pack_relu_config(const ckernel::ReluConfig& relu_config) {
    _llk_pack_relu_config_<p_pacr::PACK0, false /* EN_32B_DEST */>(relu_config);
}

/**
 * @brief: Configure packer0 to enable or disable l1 accumulation
 * @param l1_acc_en: if false -> l1 acc is disabled, true -> l1 acc enabled
 **/
inline void llk_pack_reconfig_l1_acc(const std::uint32_t l1_acc_en) { _llk_pack_set_l1_acc_<p_pacr::PACK0>(l1_acc_en); }
