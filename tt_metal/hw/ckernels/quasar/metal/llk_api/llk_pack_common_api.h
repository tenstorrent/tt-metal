// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "llk_outputs.h"
#include "llk_pack_common.h"
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

    // Program buffer descriptors for all 32 dataflow buffers, i is the logical dfb id
    for (std::uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS; ++i) {
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

        ckernel::trisc::_configure_buf_desc_table_(i, bd_val);
    }

    tdma_descriptor_t td_val;
    td_val.reg_data_format = static_cast<std::uint8_t>(pack_src_format[output_id]);

    _llk_pack_hw_configure_<p_pacr::PACK0>(td_val);
}

/**
 * @brief Clears the data valid for destination register after Packer 0 is done packing
 * and zeroes out the dest bank(s) used by packer 0
 *
 * @tparam DST: Destination register buffering mode, values = [DstSync::SyncHalf, DstSync::SyncFull]
 * @tparam IS_FP32_MATH_DEST_EN: flag to show if math destination register is set to float32 mode
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
 */
inline void llk_packer_wait_for_math_done() { _llk_packer_wait_for_math_done_(); }

/**
 * @brief Signals that the packer has finished consuming the current Destination Register section.
 * Posts to the math–pack semaphore and clears/zeros the dest bank(s) used by the packer;
 *
 * @tparam is_fp32_dest_acc_en True if math destination is in 32-bit mode, false for 16-bit mode.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_pack_dest_section_done() {
    _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, DST_SYNC_MODE, is_fp32_dest_acc_en>();
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
