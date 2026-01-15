// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "api/debug/waypoint.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_param_structs.h"
#include "llk_unpack_common.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

template <bool is_fp32_dest_acc_en, bool disable_src_zero_flag = false>
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {
    // In0 -> unpA
    // In1 -> unpB
    const uint32_t unpA_operand_id = get_operand_id(unpA_operand);
    const uint32_t unpB_operand_id = get_operand_id(unpB_operand);

    // unpA -> srcA
    // unpB -> srcB
    const uint32_t unpA_num_faces = get_operand_num_faces(unpA_operand_id);
    const uint32_t unpA_face_r_dim = get_operand_face_r_dim(unpA_operand_id);
    const uint32_t unpB_num_faces = get_operand_num_faces(unpB_operand_id);
    const uint32_t unpB_face_r_dim = get_operand_face_r_dim(unpB_operand_id);

    // Currently, there is a constraint that tile size is equal to the fifo page size
    // TODO NC: tile size should be computed in the LLK instead, as the part of #34495
    const uint32_t unpA_tile_size = get_local_cb_interface(unpA_operand_id).fifo_page_size;
    const uint32_t unpB_tile_size = get_local_cb_interface(unpB_operand_id).fifo_page_size;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en, disable_src_zero_flag>(
        unpack_src_format[unpA_operand_id],
        unpack_src_format[unpB_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpack_dst_format[unpB_operand_id],
        unpA_face_r_dim,
        unpB_face_r_dim,
        unpA_num_faces,
        unpB_num_faces,
        unpA_tile_size,
        unpB_tile_size);
}

template <bool is_fp32_dest_acc_en, bool disable_src_zero_flag = false>
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand) {
    llk_unpack_hw_configure<is_fp32_dest_acc_en, disable_src_zero_flag>(unpA_operand, unpA_operand);
}

inline bool should_reconfigure_cbs(std::uint32_t old_operand, std::uint32_t new_operand) {
    return (unpack_src_format[old_operand] != unpack_src_format[new_operand]) ||
           (unpack_dst_format[old_operand] != unpack_dst_format[new_operand]);
}

// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    const std::uint32_t srca_operand_id = get_operand_id(srca_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srca_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srca_operand_id);

    // Currently, there is a constraint that tile size is equal to the fifo page size
    // TODO NC: tile size should be computed in the LLK instead, as the part of #34495
    const std::uint32_t tile_size = get_local_cb_interface(srca_operand_id).fifo_page_size;
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, to_from_int8>(
        unpack_src_format[srca_operand_id], unpack_dst_format[srca_operand_id], tile_size);
}

// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    std::uint32_t srcb_operand_id = get_operand_id(srcb_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srcb_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srcb_operand_id);

    // Currently, there is a constraint that tile size is equal to the fifo page size
    // TODO NC: tile size should be computed in the LLK instead, as the part of #34495
    const std::uint32_t tile_size = get_local_cb_interface(srcb_operand_id).fifo_page_size;
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, to_from_int8>(
        unpack_src_format[srcb_operand_id], unpack_dst_format[srcb_operand_id], tile_size);
}

// TODO NC: Clean up as the part of tt-metal#34499
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

// TODO NC: Clean up as the part of tt-metal#34499
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

// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8, is_tile_dim_reconfig_en>(srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8, is_tile_dim_reconfig_en>(srcb_new_operand);
}

// TODO NC: Clean up as the part of tt-metal#34499
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

inline void llk_unpack_set_srcb_dummy_valid() { _llk_unpack_set_srcb_dummy_valid_(); }
