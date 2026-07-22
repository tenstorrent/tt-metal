// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
 * Configure the unpacker hardware for operands A and B.
 *
 * Face geometry (face_r_dim, num_faces) and tile size for both operands are
 * derived from the CB metadata associated with each operand id. This is the
 * primary entry point: callers no longer need to thread face geometry through
 * the API, since per-CB face geometry is recorded in the CB descriptor at
 * program creation time.
 *
 * @tparam is_fp32_dest_acc_en   Enable FP32 accumulation in the destination register.
 * @param  unpA_operand          Operand index for unpack source A (In0).
 * @param  unpB_operand          Operand index for unpack source B (In1).
 */
template <bool is_fp32_dest_acc_en>
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

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
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
 * Single-operand convenience overload that configures both unpack sources from
 * the same operand id. Equivalent to calling the two-operand overload with
 * unpA_operand == unpB_operand.
 *
 * @tparam is_fp32_dest_acc_en   Enable FP32 accumulation in the destination register.
 * @param  unpA_operand          Operand index used for both unpack source A and B.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand) {
    llk_unpack_hw_configure<is_fp32_dest_acc_en>(unpA_operand, unpA_operand);
}

/**
 * Determine whether the unpacker must be reconfigured when switching operands, i.e. whether the
 * source or destination data format differs between the two operands.
 *
 * @param old_operand Currently configured operand id.
 * @param new_operand Candidate operand id to switch to.
 * @return True if the src or dst data format differs between the operands.
 */
inline bool should_reconfigure_cbs(std::uint32_t old_operand, std::uint32_t new_operand) {
    return (unpack_src_format[old_operand] != unpack_src_format[new_operand]) ||
           (unpack_dst_format[old_operand] != unpack_dst_format[new_operand]);
}

/**
 * Reconfigure the srcA unpacker for a new operand's data format.
 *
 * Face geometry (face_r_dim, num_faces) and tile size are derived from the new operand's CB metadata.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam to_from_int8        Whether the reconfiguration crosses an int8 format boundary.
 * @param  srca_new_operand    New operand id to configure srcA for.
 */
// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    const std::uint32_t srca_operand_id = get_operand_id(srca_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srca_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srca_operand_id);

    // Currently, there is a constraint that tile size is equal to the fifo page size
    // TODO NC: tile size should be computed in the LLK instead, as the part of #34495
    const std::uint32_t tile_size = get_local_cb_interface(srca_operand_id).fifo_page_size;
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(
        unpack_src_format[srca_operand_id], unpack_dst_format[srca_operand_id], tile_size, face_r_dim, num_faces);
}

/**
 * Reconfigure the srcB unpacker for a new operand's data format.
 *
 * Face geometry (face_r_dim, num_faces) and tile size are derived from the new operand's CB metadata.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam to_from_int8        Whether the reconfiguration crosses an int8 format boundary.
 * @param  srcb_new_operand    New operand id to configure srcB for.
 */
// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    std::uint32_t srcb_operand_id = get_operand_id(srcb_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srcb_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srcb_operand_id);

    // Currently, there is a constraint that tile size is equal to the fifo page size
    // TODO NC: tile size should be computed in the LLK instead, as the part of #34495
    const std::uint32_t tile_size = get_local_cb_interface(srcb_operand_id).fifo_page_size;
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(
        unpack_src_format[srcb_operand_id], unpack_dst_format[srcb_operand_id], tile_size, face_r_dim, num_faces);
}

/**
 * Conditionally reconfigure the srcA unpacker when switching operands. Reprograms only when the CBs
 * differ, an explicit dim/stride target is requested, or the face geometry changed between the old
 * and new operands.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam to_from_int8        Whether the reconfiguration crosses an int8 format boundary.
 * @param  srca_old_operand    Currently configured srcA operand id.
 * @param  srca_new_operand    New srcA operand id to switch to.
 */
// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if (should_reconfigure_cbs(srca_old_operand, srca_new_operand)) {
        llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(srca_new_operand);
    } else if constexpr (dim_stride_target != p_dim_stride_target::IGNORE) {
        llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(srca_new_operand);
    } else if (
        get_operand_face_r_dim(old_srca_operand_id) != get_operand_face_r_dim(new_srca_operand_id) ||
        get_operand_num_faces(old_srca_operand_id) != get_operand_num_faces(new_srca_operand_id)) {
        llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, to_from_int8>(
            srca_new_operand);
    }
}

/**
 * Conditionally reconfigure the srcB unpacker when switching operands. Reprograms only when the CBs
 * differ, an explicit dim/stride target is requested, or the face geometry changed between the old
 * and new operands.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam to_from_int8        Whether the reconfiguration crosses an int8 format boundary.
 * @param  srcb_old_operand    Currently configured srcB operand id.
 * @param  srcb_new_operand    New srcB operand id to switch to.
 */
// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if (should_reconfigure_cbs(srcb_old_operand, srcb_new_operand)) {
        llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(srcb_new_operand);
    } else if constexpr (dim_stride_target != p_dim_stride_target::IGNORE) {
        llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(srcb_new_operand);
    } else if (
        get_operand_face_r_dim(old_srcb_operand_id) != get_operand_face_r_dim(new_srcb_operand_id) ||
        get_operand_num_faces(old_srcb_operand_id) != get_operand_num_faces(new_srcb_operand_id)) {
        llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, to_from_int8>(
            srcb_new_operand);
    }
}

/**
 * Reconfigure both srcA and srcB unpackers for new operands and refresh the zero-source flag.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam to_from_int8        Whether the reconfiguration crosses an int8 format boundary.
 * @param  srca_new_operand    New srcA operand id.
 * @param  srcb_new_operand    New srcB operand id.
 */
// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(srcb_new_operand);
}

/**
 * Conditionally reconfigure both srcA and srcB unpackers when switching operands (using the old
 * operands to decide whether reprogramming is needed) and refresh the zero-source flag.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam to_from_int8        Whether the reconfiguration crosses an int8 format boundary.
 * @param  srca_old_operand    Currently configured srcA operand id.
 * @param  srca_new_operand    New srcA operand id.
 * @param  srcb_old_operand    Currently configured srcB operand id.
 * @param  srcb_new_operand    New srcB operand id.
 */
// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(
        srca_old_operand, srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(
        srcb_old_operand, srcb_new_operand);
}

/**
 * Mark srcB as holding dummy-valid data so the math thread can proceed without a real srcB unpack.
 */
inline void llk_unpack_set_srcb_dummy_valid() { _llk_unpack_set_srcb_dummy_valid_(); }
