// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "api/debug/waypoint.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_unpack_common.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

/**
 * @brief Programs l1 info & source register format for both UNP_A and UNP_B
 *
 * @param operandA: The input0 operand circular buffer
 * @param operandB: The input1 operand circular buffer
 */
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {
    const std::uint32_t unpA_operand_id = get_operand_id(unpA_operand);
    const std::uint32_t unpB_operand_id = get_operand_id(unpB_operand);

    // Program buffer descriptors for all 32 dataflow buffers, i is the logical dfb id.
    // Skip non-participating DFBs via entry_size==0 (g_dfb_interface[] is zero-init,
    // so non-populated entries naturally fall out). Loop bound is dfb::NUM_DFBS because
    // g_dfb_interface[] is sized NUM_DFBS (=32) and NUM_CIRCULAR_BUFFERS resolves to 64
    // on Quasar — GCC -Werror=aggressive-loop-optimizations rejects the OOB.
    for (std::uint32_t i = 0; i < dfb::NUM_DFBS; ++i) {
        if (g_dfb_interface[i].entry_size == 0) {
            continue;
        }
        const DataFormat l1_data_format = static_cast<DataFormat>(unpack_src_format[i]);

        if (l1_data_format == DataFormat::Invalid) {
            continue;
        }

        // TODO: with multiple TCs are there multiple descriptors?
        buffer_descriptor_u bd_val = {0};
        bd_val.f.l1_addr_16B = get_local_dfb_interface(i).tc_slots[0].base_addr;
        bd_val.f.format = static_cast<std::uint8_t>(l1_data_format);
        bd_val.f.x_dim = ckernel::trisc::FACE_C_DIM;
        bd_val.f.y_dim = unpack_tile_face_r_dim[i];
        bd_val.f.z_dim = unpack_tile_num_faces[i];

        ckernel::trisc::_configure_buf_desc_table_(i, bd_val);
    }

    tdma_descriptor_t td_val_A, td_val_B;
    td_val_A.reg_data_format = static_cast<std::uint8_t>(unpack_dst_format[unpA_operand_id]);
    td_val_B.reg_data_format = static_cast<std::uint8_t>(unpack_dst_format[unpB_operand_id]);

    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);
}

/**
 * @brief Programs l1 info & source register format for UNP_A
 *
 * @param operandA: The input operand circular buffer
 */
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand) {
    llk_unpack_hw_configure(unpA_operand, unpA_operand);
}

inline bool should_reconfig_src_reg_df(std::uint32_t old_operand, std::uint32_t new_operand) {
    return (unpack_src_format[old_operand] != unpack_src_format[new_operand]) ||
           (unpack_dst_format[old_operand] != unpack_dst_format[new_operand]);
}

/**
 * Reprograms unpacker THCON OUT_DATA_FORMAT only (gasket); L1 format stays in buffer descriptors.
 */
template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, [[maybe_unused]] bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    const std::uint32_t srca_operand_id = get_operand_id(srca_new_operand);
    _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_A, EN_32BIT_DEST>(
        unpack_src_format[srca_operand_id], unpack_dst_format[srca_operand_id]);
}

template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, [[maybe_unused]] bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    const std::uint32_t srcb_operand_id = get_operand_id(srcb_new_operand);
    _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_B, EN_32BIT_DEST>(
        unpack_src_format[srcb_operand_id], unpack_dst_format[srcb_operand_id]);
}

template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    // Silent no-op if old/new operands already share both src and dst formats.
    if (!should_reconfig_src_reg_df(srca_old_operand, srca_new_operand)) {
        return;
    }
    llk_unpack_reconfig_data_format_srca<EN_32BIT_DEST, dim_stride_target, to_from_int8>(srca_new_operand);
}

template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    // Silent no-op if old/new operands already share both src and dst formats.
    if (!should_reconfig_src_reg_df(srcb_old_operand, srcb_new_operand)) {
        return;
    }
    llk_unpack_reconfig_data_format_srcb<EN_32BIT_DEST, dim_stride_target, to_from_int8>(srcb_new_operand);
}

template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    llk_unpack_reconfig_data_format_srca<EN_32BIT_DEST, dim_stride_target, to_from_int8>(srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<EN_32BIT_DEST, dim_stride_target, to_from_int8>(srcb_new_operand);
}

template <bool EN_32BIT_DEST, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    static_assert(
        dim_stride_target == p_dim_stride_target::IGNORE,
        "Quasar unpack reconfig does not support stride/tile-dimension changes");
    llk_unpack_reconfig_data_format_srca<EN_32BIT_DEST, dim_stride_target, to_from_int8>(
        srca_old_operand, srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<EN_32BIT_DEST, dim_stride_target, to_from_int8>(
        srcb_old_operand, srcb_new_operand);
}
