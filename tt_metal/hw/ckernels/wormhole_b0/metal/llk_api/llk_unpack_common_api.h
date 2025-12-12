// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "circular_buffer.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_param_structs.h"
#include "llk_unpack_common.h"
#include "debug/waypoint.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

inline bool should_reconfigure_cbs(std::uint32_t old_operand, std::uint32_t new_operand) {
    return (unpack_src_format[old_operand] != unpack_src_format[new_operand]) ||
           (unpack_dst_format[old_operand] != unpack_dst_format[new_operand]);
}

void llk_zero_operand(std::uint32_t operand) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t fifo_base_addr =
        (get_local_cb_interface(operand_id).fifo_limit + 1) - get_local_cb_interface(operand_id).fifo_size;
    std::uint32_t size = get_local_cb_interface(operand_id).fifo_size;
    _llk_zero_buffer_(fifo_base_addr, size);
}

template <bool mail2math = true, bool mail2pack = true>
inline void llk_unpack_get_tile(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t* p_tile) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    std::uint32_t address = base_address + offset_address;
    _llk_unpack_get_tile_<mail2math, mail2pack>(address, p_tile);
}

template <bool mail2math = true, bool mail2pack = true>
inline void llk_unpack_release_tile(std::uint32_t operand) {
    _llk_unpack_release_tile_<mail2math, mail2pack>();
}

inline void llk_unpack_debug_dump(std::uint8_t* data, std::uint32_t byte_size) {
    _llk_unpack_debug_dump_(data, byte_size);
}

inline void llk_unpack_debug_dump_seek(std::uint8_t offset) { _llk_unpack_debug_dump_seek_(offset); }

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    const std::uint32_t srca_operand_id = get_operand_id(srca_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srca_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srca_operand_id);
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, to_from_int8>(
        unpack_src_format[srca_operand_id],
        unpack_dst_format[srca_operand_id],
        get_local_cb_interface(srca_operand_id).fifo_page_size);
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    std::uint32_t srcb_operand_id = get_operand_id(srcb_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srcb_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srcb_operand_id);
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, to_from_int8>(
        unpack_src_format[srcb_operand_id],
        unpack_dst_format[srcb_operand_id],
        get_local_cb_interface(srcb_operand_id).fifo_page_size);
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if (should_reconfigure_cbs(srca_old_operand, srca_new_operand)) {
        llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8, is_tile_dim_reconfig_en>(
            srca_new_operand);
    } else if constexpr (is_tile_dim_reconfig_en) {
        llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8, is_tile_dim_reconfig_en>(
            srca_new_operand);
    }
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if (should_reconfigure_cbs(srcb_old_operand, srcb_new_operand)) {
        llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8, is_tile_dim_reconfig_en>(
            srcb_new_operand);
    } else if constexpr (is_tile_dim_reconfig_en) {
        llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8, is_tile_dim_reconfig_en>(
            srcb_new_operand);
    }
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8, is_tile_dim_reconfig_en>(srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8, is_tile_dim_reconfig_en>(srcb_new_operand);
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8, is_tile_dim_reconfig_en>(
        srca_old_operand, srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8, is_tile_dim_reconfig_en>(
        srcb_old_operand, srcb_new_operand);
}

inline void llk_unpack_dbg_feature_disable() { _llk_unpack_dbg_feature_disable_(); }
inline void llk_unpack_clear_dbg_feature_disable() { _llk_unpack_clear_dbg_feature_disable_(); }

inline void llk_enable_int8_fpu_math() { _llk_enable_int8_fpu_math_(); }

inline void llk_unpack_set_srcb_dummy_valid() { _llk_unpack_set_srcb_dummy_valid_(); }

// All TILE_SIZE related functions were deprecared in BBE for WH.  The following is needed for pack_shifted so just
// keeping here.
// FIXME: Need to review and adjust accordingly
constexpr static std::int32_t MUL_HEADERLESS_TILE_SIZE_AND_INDEX(uint format, uint index) {
    switch (format & 0x1F) {
        case ((uint8_t)DataFormat::Float32): return ((index << 8));
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b): return ((index << 7));
        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b): return ((index << 6) + (index << 2));
        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((index << 5) + (index << 2));
        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((index << 4) + (index << 2));
        case ((uint8_t)DataFormat::Int8):
        case ((uint8_t)DataFormat::Lf8): return ((index << 6));
        // Keep default as Bfp8?
        default: return ((index << 6) + (index << 2));
    };
}
