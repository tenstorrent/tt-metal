// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ckernel.h"
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
 * @param pack_output The output DataFlow Buffer identifier
 */
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
        buffer_descriptor_u bd_val = {0};
        bd_val.f.l1_addr_16B = get_local_dfb_interface(i).tc_slots[0].base_addr;
        bd_val.f.format = static_cast<std::uint8_t>(l1_data_format);
        bd_val.f.x_dim = ckernel::trisc::FACE_C_DIM;
        bd_val.f.y_dim = pack_tile_face_r_dim[i];
        bd_val.f.z_dim = pack_tile_num_faces[i];

        ckernel::trisc::validate_buffer_desc<ckernel::trisc::L1AccessMode::Continuous>(bd_val);
        ckernel::trisc::_configure_buf_desc_table_(i, bd_val);
    }

    tdma_descriptor_t td_val;
    td_val.reg_data_format = static_cast<std::uint8_t>(pack_src_format[output_id]);

    _llk_pack_hw_configure_<p_pacr::PACK0>(td_val);
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
 * @tparam DST: Destination register buffering mode, values = [DstSync::SyncHalf, DstSync::SyncFull]
 * @tparam IS_FP32_MATH_DEST_EN: flag to show if math destination register is set to float32 mode
 *
 * @warning SYNC SCHEME: dest-dvalid. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 **/
template <DstSync DST, bool IS_FP32_MATH_DEST_EN>
inline void llk_pack_dest_dvalid_section_done() {
    _llk_pack_dest_dvalid_section_done_<DST, IS_FP32_MATH_DEST_EN>();
}

/**
 * All the following functions are added to enable Math <-> Pack synchronization
 * on the destination register using semaphores.
 *
 * The following functions should be phased out once the dest dvalid scheme is introduced
 */
// TODO: AM; move from semaphores to a per op programmable dest dvalid scheme, issue #37468

/**
 * @brief Waits until math has finished producing data for the current Destination Register section.
 * Blocks on the math–pack semaphore so the packer does not read dest before math has written it.
 *
 * @warning SYNC SCHEME: semaphores. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 */
inline void llk_packer_wait_for_math_done() { _llk_packer_wait_for_math_done_(); }

/**
 * @brief Signals that the packer has finished consuming the current Destination Register section.
 * Posts to the math–pack semaphore and clears/zeros the dest bank(s) used by the packer;
 *
 * @tparam EN_32BIT_DEST True if math destination is in 32-bit mode, false for 16-bit mode.
 *
 * @warning SYNC SCHEME: semaphores. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 */
template <bool EN_32BIT_DEST>
inline void llk_pack_dest_section_done() {
    if constexpr (UnpackToDestEn) {
        _llk_sync_get_<p_stall::PACK0>(semaphore::MATH_PACK);
        if constexpr (DST_SYNC_MODE == DstSync::SyncHalf) {
            _llk_sync_advance_dest_section_<ckernel::TRISC_ID, true /*EN_32BIT_DEST*/, p_stall::PACK0>();
        }
    } else {
        _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, DST_SYNC_MODE, EN_32BIT_DEST>();
    }
}

/**
 * @brief Reset packer dest-bank parity to bank 0 at program start (pack-side mirror of llk_math_pack_sync_init).
 *
 * @warning SYNC SCHEME: semaphores. There are two mutually exclusive Dest register synchronization schemes: the
 * dest-dvalid scheme and the semaphore scheme. Never mix them. Currently the semaphore scheme is used in llk and
 * compute APIs.
 */
inline void llk_pack_dest_init() { _llk_pack_dest_init_<p_pacr::PACK0, DST_SYNC_MODE>(); }

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
