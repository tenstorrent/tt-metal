// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ckernel.h"
#include "llk_dest_dvalid.h"
#include "llk_outputs.h"
#include "llk_pack_common.h"
#include "llk_sync.h"
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
 * The destination register is shared between UNPACK, MATH and PACK through the dest data-valid chain:
 * each client waits for its own dvalid bit, then toggles it and its successor's on hand-off. PACK is
 * always the last client, so unlike the unpack and math slots its programming does not depend on the op
 * and is done once, in @ref llk_pack_dest_init.
 */

/**
 * @brief Waits until math has finished producing data for the current Destination Register section.
 *
 * Empty by design: the pack client's DEST_DVALID_CTRL wait mask gates the packer's DEST accesses in
 * hardware, so there is nothing for software to block on.
 *
 * @note Pair with @ref llk_pack_dest_section_done, and on the math thread with @ref llk_math_wait_for_dest_available.
 */
inline void llk_packer_wait_for_math_done() {}

/**
 * @brief Signals that the packer has finished consuming the current Destination Register section.
 *
 * Zeroes the dest bank the packer just read and releases the section back to the head of the chain.
 *
 * @tparam EN_32BIT_DEST True if math destination is in 32-bit mode, false for 16-bit mode.
 * @note Pair with @ref llk_packer_wait_for_math_done.
 */
template <bool EN_32BIT_DEST>
inline void llk_pack_dest_section_done() {
    _llk_dest_dvalid_signal_<dest_dvalid::client::PACK, DST_SYNC_MODE, EN_32BIT_DEST>();
}

/**
 * @brief Reset packer dest-bank parity to bank 0 and program the packer's slot in the dest data-valid chain.
 *
 * @note Call once per program, before the first pack; pair with @ref llk_math_pack_sync_init (T1).
 */
inline void llk_pack_dest_init() {
    _llk_pack_dest_init_<p_pacr::PACK0, DST_SYNC_MODE>();
    _llk_dest_dvalid_configure_<dest_dvalid::client::PACK, dest_dvalid::client::PACK>();
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
