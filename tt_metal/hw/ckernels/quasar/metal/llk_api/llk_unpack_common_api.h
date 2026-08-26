// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "api/debug/waypoint.h"
#include "llk_bfd_alloc.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_unpack_common.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

/**
 * @brief Allocate a BFD id from the unpack partition, program its table entry from the operand's
 * DFB info (shape, L1 base, formats), and record the id in the engine's current slot. The DFB id is
 * used only to fetch buffer info — it never doubles as the BFD id.
 *
 * @tparam E: physical UNPACR engine that will consume the descriptor (Unp0 or Unp1)
 * @tparam MODE: L1 access mode for the descriptor; Strided collapses y/z dims to 1 for the
 * UNPACR_STRIDE tilize sequences.
 */
template <ckernel::trisc::BfdResource E, ckernel::trisc::L1AccessMode MODE = ckernel::trisc::L1AccessMode::Continuous>
inline void llk_unpack_program_bfd(const std::uint32_t operand_id) {
    // TODO: multi-TC not handled — only tc_slots[0]'s L1 base is programmed. When a DFB is mapped
    // across multiple TCs this must program one descriptor per active tc_slot (same gap in
    // llk_pack_program_bfd). Tied to the DFB<->buffer-descriptor decouple work.
    ckernel::trisc::bfd_alloc_and_program<E, MODE>(
        get_operand_tensor_shape(operand_id),
        get_local_dfb_interface(operand_id).tc_slots[0].base_addr,
        static_cast<std::uint32_t>(unpack_src_format[operand_id]));
}

/**
 * @brief Programs source register format for both UNP_A and UNP_B
 *
 * Buffer descriptors are no longer programmed here: BFD ids are an LLK-internal resource
 * allocated from the per-TRISC partition (see llk_bfd_alloc.h) and each op's llk_unpack_*_init
 * programs its own table entry. DFB ids never double as BFD ids.
 *
 * @param operandA: The input0 operand circular buffer
 * @param operandB: The input1 operand circular buffer
 */
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {
    const std::uint32_t unpA_operand_id = get_operand_id(unpA_operand);
    const std::uint32_t unpB_operand_id = get_operand_id(unpB_operand);

    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
        static_cast<DataFormat>(unpack_dst_format[unpA_operand_id]),
        static_cast<DataFormat>(unpack_dst_format[unpB_operand_id]));
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

/**
 * @brief Issues a dummy SrcB dvalid so the math thread can satisfy its SRCB_VLD
 * stall in transpose-dest. Used by the transpose_wh_dest compute API.
 */
inline void llk_unpack_set_srcb_dummy_valid() { _llk_unpack_set_srcB_dummy_valid_(); }
