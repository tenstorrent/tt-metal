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
#include "api/debug/waypoint.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_param_structs.h"
#include "llk_unpack_common.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

/**
 * @brief Configure the unpacker hardware for both operands A and B.
 *
 * Face geometry (face_r_dim, num_faces) and tile size for both operands are derived from the CB
 * metadata associated with each operand id, so callers no longer thread face geometry through the
 * API.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam disable_src_zero_flag: Disable the source zero-substitution flag.
 * @param unpA_operand: Operand index for unpack source A (In0).
 * @param unpB_operand: Operand index for unpack source B (In1).
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
 * @deprecated Operand A face geometry (face_r_dim, num_faces) is now derived from the CB metadata. Use the
 * llk_unpack_hw_configure(unpA_operand, unpB_operand) overload instead. This explicit-face-geometry overload
 * is retained only for backwards compatibility and will be removed.
 *
 * @tparam is_fp32_dest_acc_en   Enable FP32 accumulation in the destination register.
 * @tparam disable_src_zero_flag Disable the source zero flag.
 * @param  unpA_operand          Input operand index for unpack source A.
 * @param  unpB_operand          Input operand index for unpack source B.
 * @param  unpA_face_r_dim       Face row dimension for operand A (overrides operand metadata).
 * @param  unpA_num_faces        Number of faces for operand A (overrides operand metadata).
 */
template <bool is_fp32_dest_acc_en, bool disable_src_zero_flag = false>
[[deprecated(
    "Operand A face geometry is now derived from CB metadata; use the "
    "llk_unpack_hw_configure(unpA_operand, unpB_operand) overload instead.")]] inline void
llk_unpack_hw_configure(
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
 * @brief Configure the unpacker hardware using a single operand for both sources.
 *
 * Convenience overload equivalent to calling the two-operand overload with unpA_operand ==
 * unpB_operand.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam disable_src_zero_flag: Disable the source zero-substitution flag.
 * @param unpA_operand: Operand index used for both unpack source A and B.
 */
template <bool is_fp32_dest_acc_en, bool disable_src_zero_flag = false>
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand) {
    llk_unpack_hw_configure<is_fp32_dest_acc_en, disable_src_zero_flag>(unpA_operand, unpA_operand);
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
 * @brief Reconfigure the operand A (srcA) unpacker for a new operand's data format.
 *
 * Face geometry (face_r_dim, num_faces) and tile size are derived from the new operand's CB metadata.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam to_from_int8: Reconfiguring to/from an Int8 format (requires FP32 dest mode).
 * @param srca_new_operand: New operand index to configure srcA for.
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
 * @brief Reconfigure the operand B (srcB) unpacker for a new operand's data format.
 *
 * Face geometry (face_r_dim, num_faces) and tile size are derived from the new operand's CB metadata.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam to_from_int8: Reconfiguring to/from an Int8 format (requires FP32 dest mode).
 * @param srcb_new_operand: New operand index to configure srcB for.
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
 * @brief Conditionally reconfigure the srcA unpacker when switching operands.
 *
 * Reprograms only when the CBs differ, an explicit dim/stride target is requested, or the face
 * geometry changed between the old and new operands.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam to_from_int8: Reconfiguring to/from an Int8 format (requires FP32 dest mode).
 * @param srca_old_operand: Currently configured srcA operand index.
 * @param srca_new_operand: New srcA operand index to switch to.
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
 * @brief Conditionally reconfigure the srcB unpacker when switching operands.
 *
 * Reprograms only when the CBs differ, an explicit dim/stride target is requested, or the face
 * geometry changed between the old and new operands.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam to_from_int8: Reconfiguring to/from an Int8 format (requires FP32 dest mode).
 * @param srcb_old_operand: Currently configured srcB operand index.
 * @param srcb_new_operand: New srcB operand index to switch to.
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
 * @brief Reconfigure both srcA and srcB unpackers for new operands and refresh the zero-source flag.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam to_from_int8: Reconfiguring to/from an Int8 format (requires FP32 dest mode).
 * @param srca_new_operand: New srcA operand index.
 * @param srcb_new_operand: New srcB operand index.
 */
// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, dim_stride_target, to_from_int8>(srcb_new_operand);
    _llk_unpack_reconfig_zero_src_flag_(
        unpack_dst_format[get_operand_id(srca_new_operand)], unpack_dst_format[get_operand_id(srcb_new_operand)]);
}

/**
 * @brief Conditionally reconfigure both srcA and srcB unpackers when switching operands.
 *
 * Uses the old operands to decide whether reprogramming is needed, then refreshes the zero-source
 * flag.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam to_from_int8: Reconfiguring to/from an Int8 format (requires FP32 dest mode).
 * @param srca_old_operand: Currently configured srcA operand index.
 * @param srca_new_operand: New srcA operand index.
 * @param srcb_old_operand: Currently configured srcB operand index.
 * @param srcb_new_operand: New srcB operand index.
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
    _llk_unpack_reconfig_zero_src_flag_(
        unpack_dst_format[get_operand_id(srca_new_operand)], unpack_dst_format[get_operand_id(srcb_new_operand)]);
}

/**
 * @brief Set the unpacker debug feature-disable bit as a hardware bug workaround.
 *
 * @note Writes bit 11 of RISCV_DEBUG_REG_DBG_FEATURE_DISABLE (workaround for tt-metal#46219).
 */
// TODO NC: Remove as a part of tt-metal#36411
inline void llk_unpack_dbg_feature_disable() { _llk_unpack_dbg_feature_disable_(); }

/**
 * @brief Mark srcB as data-valid without unpacking real data.
 *
 * Lets the math thread proceed when srcB is not actually fed from L1.
 */
inline void llk_unpack_set_srcb_dummy_valid() { _llk_unpack_set_srcb_dummy_valid_(); }
