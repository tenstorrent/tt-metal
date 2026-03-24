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
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_param_structs.h"
#include "llk_unpack_common.h"
#include "api/debug/waypoint.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

/**
 * Configure the unpacker hardware with explicit face geometry for operand A.
 *
 * Unlike the two-argument overload, which derives face_r_dim and num_faces for
 * operand A from its operand metadata, this overload accepts caller-supplied
 * values for unpA_face_r_dim and unpA_num_faces. This is useful when the
 * operand A tile layout differs from what is recorded in the CB interface
 * (e.g. non-standard tile dimensions or partial-face unpacking).
 * Operand B face geometry is still derived from its operand metadata.
 *
 * @tparam is_fp32_dest_acc_en   Enable FP32 accumulation in the destination register.
 * @tparam disable_src_zero_flag When true, disables the source-zero optimisation flag.
 * @param  unpA_operand          Operand index for unpack source A (In0).
 * @param  unpB_operand          Operand index for unpack source B (In1).
 * @param  unpA_face_r_dim       Row dimension of each face for operand A (overrides operand metadata).
 * @param  unpA_num_faces        Number of faces for operand A (overrides operand metadata).
 */
template <bool is_fp32_dest_acc_en, bool disable_src_zero_flag = false>
inline void llk_unpack_hw_configure(
    const std::uint32_t unpA_operand,
    const std::uint32_t unpB_operand,
    const std::uint32_t unpA_face_r_dim,
    const std::uint32_t unpA_num_faces) {
    // In0 -> unpA
    // In1 -> unpB
    const uint32_t unpA_operand_id = get_operand_id(unpA_operand);
    const uint32_t unpB_operand_id = get_operand_id(unpB_operand);

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

/**
 * Configure the unpacker hardware for operands A (srcA) and B (srcB).
 *
 * Face geometry (face_r_dim, num_faces) for both operands is derived from
 * the operand metadata registered in the circular buffer interface.
 *
 * @tparam is_fp32_dest_acc_en   Enable FP32 accumulation in the destination register.
 * @tparam disable_src_zero_flag When true, disables the source-zero optimisation flag.
 * @param  unpA_operand          Operand index for unpack source A (In0).
 * @param  unpB_operand          Operand index for unpack source B (In1).
 */
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

    llk_unpack_hw_configure<is_fp32_dest_acc_en, disable_src_zero_flag>(
        unpA_operand, unpB_operand, unpA_face_r_dim, unpA_num_faces);
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
template <
    bool is_fp32_dest_acc_en,
    bool to_from_int8 = false,
    p_dim_stride_target dim_stride_target = p_dim_stride_target::IGNORE>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    const std::uint32_t srca_operand_id = get_operand_id(srca_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srca_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srca_operand_id);

    // Currently, there is a constraint that tile size is equal to the fifo page size
    // TODO NC: tile size should be computed in the LLK instead, as the part of #34495
    const std::uint32_t tile_size = get_local_cb_interface(srca_operand_id).fifo_page_size;
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(
        unpack_src_format[srca_operand_id], unpack_dst_format[srca_operand_id], tile_size, face_r_dim, num_faces);
}

// TODO NC: Clean up as the part of tt-metal#34499
template <
    bool is_fp32_dest_acc_en,
    bool to_from_int8 = false,
    p_dim_stride_target dim_stride_target = p_dim_stride_target::IGNORE>
inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    std::uint32_t srcb_operand_id = get_operand_id(srcb_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srcb_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srcb_operand_id);

    // Currently, there is a constraint that tile size is equal to the fifo page size
    // TODO NC: tile size should be computed in the LLK instead, as the part of #34495
    const std::uint32_t tile_size = get_local_cb_interface(srcb_operand_id).fifo_page_size;
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(
        unpack_src_format[srcb_operand_id], unpack_dst_format[srcb_operand_id], tile_size, face_r_dim, num_faces);
}

// TODO NC: Clean up as the part of tt-metal#34499
template <
    bool is_fp32_dest_acc_en,
    bool to_from_int8 = false,
    p_dim_stride_target dim_stride_target = p_dim_stride_target::IGNORE>
inline void llk_unpack_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if (should_reconfigure_cbs(srca_old_operand, srca_new_operand)) {
        llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(srca_new_operand);
    } else if constexpr (dim_stride_target != p_dim_stride_target::IGNORE) {
        llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(srca_new_operand);
    }
}

// TODO NC: Clean up as the part of tt-metal#34499
template <
    bool is_fp32_dest_acc_en,
    bool to_from_int8 = false,
    p_dim_stride_target dim_stride_target = p_dim_stride_target::IGNORE>
inline void llk_unpack_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if (should_reconfigure_cbs(srcb_old_operand, srcb_new_operand)) {
        llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(srcb_new_operand);
    } else if constexpr (dim_stride_target != p_dim_stride_target::IGNORE) {
        llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(srcb_new_operand);
    }
}

// TODO NC: Clean up as the part of tt-metal#34499
template <
    bool is_fp32_dest_acc_en,
    bool to_from_int8 = false,
    p_dim_stride_target dim_stride_target = p_dim_stride_target::IGNORE>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(srcb_new_operand);
}

// TODO NC: Clean up as the part of tt-metal#34499
template <
    bool is_fp32_dest_acc_en,
    bool to_from_int8 = false,
    p_dim_stride_target dim_stride_target = p_dim_stride_target::IGNORE>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(
        srca_old_operand, srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(
        srcb_old_operand, srcb_new_operand);
}

// TODO NC: Remove as a part of tt-metal#36411
inline void llk_unpack_dbg_feature_disable() { _llk_unpack_dbg_feature_disable_(); }

inline void llk_unpack_set_srcb_dummy_valid() { _llk_unpack_set_srcb_dummy_valid_(); }
