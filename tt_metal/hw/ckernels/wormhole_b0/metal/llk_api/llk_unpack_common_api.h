// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sanitizer/api.h"
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
#include "llk_assert.h"
#include "llk_fp32_dest_acc.h"
#include "llk_unpack_common.h"
#include "api/debug/waypoint.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

/**
 * Configure the unpacker hardware for operands A and B.
 *
 * Face geometry (face_r_dim, num_faces) for both operands is derived from the CB
 * metadata associated with each operand id; the LLK sizes the tile itself from
 * that geometry. This is the primary entry point: callers no longer need to thread
 * face geometry through the API, since per-CB face geometry is recorded in the CB
 * descriptor at program creation time.
 *
 * @tparam is_fp32_dest_acc_en   Enable FP32 accumulation in the destination register.
 * @param  unpA_operand          Operand index for unpack source A (In0).
 * @param  unpB_operand          Operand index for unpack source B (In1).
 */
/**
 * @brief Assert that an operand's CB page size matches the tile size derived from its geometry.
 *
 * @param operand_id: Operand index into the CB interface.
 * @param unpack_src_format: Source data format of the operand in L1.
 * @param face_r_dim: Rows per face.
 * @param num_faces: Number of faces in the tile.
 */
inline void assert_cb_page_size_matches_tile_size(
    [[maybe_unused]] const std::uint32_t operand_id,
    [[maybe_unused]] const std::uint32_t unpack_src_format,
    [[maybe_unused]] const std::uint32_t face_r_dim,
    [[maybe_unused]] const std::uint32_t num_faces) {
    LLK_ASSERT(
        get_local_cb_interface(operand_id).fifo_page_size == _llk_unpack_tile_size_(unpack_src_format, face_r_dim, num_faces),
        "CB page size must equal the tile size derived from its src format and face geometry");
}

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

    // The LLK derives the per-tile L1 stride from src format + face geometry. The CB page size the host
    // recorded must agree with it, otherwise tile-to-tile addressing walks the wrong stride.
    assert_cb_page_size_matches_tile_size(unpA_operand_id, unpack_src_format[unpA_operand_id], unpA_face_r_dim, unpA_num_faces);
    assert_cb_page_size_matches_tile_size(unpB_operand_id, unpack_src_format[unpB_operand_id], unpB_face_r_dim, unpB_num_faces);

    SAN_HOOK(configure(
        StateVal<Operand<Exu::Unpack>::DestWidth32>(is_fp32_dest_acc_en),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[unpA_operand_id]),
        StateVal<Operand<Exu::Unpack>::InputFormatB>(unpack_src_format[unpB_operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[unpA_operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatB>(unpack_dst_format[unpB_operand_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightA>(unpA_face_r_dim),
        StateVal<Operand<Exu::Unpack>::FaceHeightB>(unpB_face_r_dim),
        StateVal<Operand<Exu::Unpack>::NumFacesA>(unpA_num_faces),
        StateVal<Operand<Exu::Unpack>::NumFacesB>(unpB_num_faces)));
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        unpack_src_format[unpA_operand_id],
        unpack_src_format[unpB_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpack_dst_format[unpB_operand_id],
        unpA_face_r_dim,
        unpB_face_r_dim,
        unpA_num_faces,
        unpB_num_faces);
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
 * Unpack-thread half of a mid-kernel FP32 dest-acc reconfiguration.
 *
 * Drains the unpacker FIFO, waits for MATH to program dest-acc CFG, then STALLWAITs.
 *
 * @note Must be called together with llk_math_set_fp32_dest_acc and llk_pack_wait_fp32_dest_acc.
 */
inline void llk_unpack_wait_fp32_dest_acc() {
    SAN_HOOK(unsupported());
    _llk_set_fp32_dest_acc_<ThreadId::UnpackThreadId>();
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
 * Face geometry (face_r_dim, num_faces) is derived from the new operand's CB metadata;
 * the LLK sizes the tile itself from that geometry.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam skip_int8           Skip re-deriving the SrcUnsigned bit from the new format; leave false unless the
 *                             caller guarantees no Int8/UInt8/Int32 boundary is crossed (tt-metal#34499).
 * @param  srca_new_operand    New operand id to configure srcA for.
 */
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    const std::uint32_t srca_operand_id = get_operand_id(srca_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srca_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srca_operand_id);

    // The LLK re-derives the per-tile L1 stride from the new src format + face geometry; the CB page size
    // the host recorded must agree with it.
    assert_cb_page_size_matches_tile_size(srca_operand_id, unpack_src_format[srca_operand_id], face_r_dim, num_faces);
    SAN_HOOK(reconfigure(
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[srca_operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[srca_operand_id]),
        StateDiscard<std::uint32_t>(face_r_dim),
        StateDiscard<std::uint32_t>(num_faces)));
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(
        unpack_src_format[srca_operand_id], unpack_dst_format[srca_operand_id], face_r_dim, num_faces);
}

/**
 * Reconfigure the srcB unpacker for a new operand's data format.
 *
 * Face geometry (face_r_dim, num_faces) is derived from the new operand's CB metadata;
 * the LLK sizes the tile itself from that geometry.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam skip_int8           Skip re-deriving the SrcUnsigned bit from the new format; leave false unless the
 *                             caller guarantees no Int8/UInt8/Int32 boundary is crossed (tt-metal#34499).
 * @param  srcb_new_operand    New operand id to configure srcB for.
 */
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    std::uint32_t srcb_operand_id = get_operand_id(srcb_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srcb_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srcb_operand_id);

    // The LLK re-derives the per-tile L1 stride from the new src format + face geometry; the CB page size
    // the host recorded must agree with it.
    assert_cb_page_size_matches_tile_size(srcb_operand_id, unpack_src_format[srcb_operand_id], face_r_dim, num_faces);
    SAN_HOOK(reconfigure(
        StateVal<Operand<Exu::Unpack>::InputFormatB>(unpack_src_format[srcb_operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatB>(unpack_dst_format[srcb_operand_id]),
        StateDiscard<std::uint32_t>(face_r_dim),
        StateDiscard<std::uint32_t>(num_faces)));
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(
        unpack_src_format[srcb_operand_id], unpack_dst_format[srcb_operand_id], face_r_dim, num_faces);
}

/**
 * Conditionally reconfigure the srcA unpacker when switching operands. Reprograms only when the CBs
 * differ, an explicit dim/stride target is requested, or the face geometry changed between the old
 * and new operands.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam skip_int8           Skip re-deriving the SrcUnsigned bit from the new format; leave false unless the
 *                             caller guarantees no Int8/UInt8/Int32 boundary is crossed (tt-metal#34499).
 * @param  srca_old_operand    Currently configured srcA operand id.
 * @param  srca_new_operand    New srcA operand id to switch to.
 */
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
inline void llk_unpack_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if (should_reconfigure_cbs(srca_old_operand, srca_new_operand)) {
        llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(srca_new_operand);
    } else if constexpr (dim_stride_target != p_dim_stride_target::IGNORE) {
        llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(srca_new_operand);
    } else if (
        get_operand_face_r_dim(old_srca_operand_id) != get_operand_face_r_dim(new_srca_operand_id) ||
        get_operand_num_faces(old_srca_operand_id) != get_operand_num_faces(new_srca_operand_id)) {
        llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, skip_int8>(
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
 * @tparam skip_int8           Skip re-deriving the SrcUnsigned bit from the new format; leave false unless the
 *                             caller guarantees no Int8/UInt8/Int32 boundary is crossed (tt-metal#34499).
 * @param  srcb_old_operand    Currently configured srcB operand id.
 * @param  srcb_new_operand    New srcB operand id to switch to.
 */
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
inline void llk_unpack_reconfig_data_format_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if (should_reconfigure_cbs(srcb_old_operand, srcb_new_operand)) {
        llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(srcb_new_operand);
    } else if constexpr (dim_stride_target != p_dim_stride_target::IGNORE) {
        llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(srcb_new_operand);
    } else if (
        get_operand_face_r_dim(old_srcb_operand_id) != get_operand_face_r_dim(new_srcb_operand_id) ||
        get_operand_num_faces(old_srcb_operand_id) != get_operand_num_faces(new_srcb_operand_id)) {
        llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, skip_int8>(
            srcb_new_operand);
    }
}

/**
 * Reconfigure both srcA and srcB unpackers for new operands and refresh the zero-source flag.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam skip_int8           Skip re-deriving the SrcUnsigned bit from the new format; leave false unless the
 *                             caller guarantees no Int8/UInt8/Int32 boundary is crossed (tt-metal#34499).
 * @param  srca_new_operand    New srcA operand id.
 * @param  srcb_new_operand    New srcB operand id.
 */
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(srcb_new_operand);
}

/**
 * Conditionally reconfigure both srcA and srcB unpackers when switching operands (using the old
 * operands to decide whether reprogramming is needed) and refresh the zero-source flag.
 *
 * @tparam is_fp32_dest_acc_en Enable FP32 accumulation in the destination register.
 * @tparam dim_stride_target   Dimension/stride programming target for the unpacker.
 * @tparam skip_int8           Skip re-deriving the SrcUnsigned bit from the new format; leave false unless the
 *                             caller guarantees no Int8/UInt8/Int32 boundary is crossed (tt-metal#34499).
 * @param  srca_old_operand    Currently configured srcA operand id.
 * @param  srca_new_operand    New srcA operand id.
 * @param  srcb_old_operand    Currently configured srcB operand id.
 * @param  srcb_new_operand    New srcB operand id.
 */
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(
        srca_old_operand, srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(
        srcb_old_operand, srcb_new_operand);
}

/**
 * Reprogram only the srcA unpacker tile/face geometry (dim/stride) for a new operand, leaving the data
 * format untouched. Face geometry (face_r_dim, num_faces) and the current dst format (for the stride
 * baselines) are read from the operand's CB metadata.
 *
 * @param srca_new_operand New operand id whose tile shape to program srcA for.
 */
inline void llk_unpack_reconfig_tile_shape_srca(const std::uint32_t srca_new_operand) {
    SAN_HOOK(unsupported());
    const std::uint32_t srca_operand_id = get_operand_id(srca_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srca_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srca_operand_id);
    // A tile-shape change is a tile-size change too; the LLK refreshes the tile-size GPR alongside geometry.
    assert_cb_page_size_matches_tile_size(srca_operand_id, unpack_src_format[srca_operand_id], face_r_dim, num_faces);
    _llk_unpack_reconfig_tile_shape_srca_(unpack_src_format[srca_operand_id], face_r_dim, num_faces);
}

/**
 * Conditionally reprogram the srcA unpacker tile/face geometry when switching operands. Reprograms only
 * when the CBs differ or the face geometry changed between the old and new operands.
 *
 * @param srca_old_operand Currently configured srcA operand id.
 * @param srca_new_operand New srcA operand id to switch to.
 */
inline void llk_unpack_reconfig_tile_shape_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    SAN_HOOK(unsupported());
    const std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    const std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if (should_reconfigure_cbs(srca_old_operand, srca_new_operand) ||
        get_operand_face_r_dim(old_srca_operand_id) != get_operand_face_r_dim(new_srca_operand_id) ||
        get_operand_num_faces(old_srca_operand_id) != get_operand_num_faces(new_srca_operand_id)) {
        llk_unpack_reconfig_tile_shape_srca(srca_new_operand);
    }
}

/**
 * Reprogram only the srcB unpacker tile/face geometry (dim/stride) for a new operand, leaving the data
 * format untouched. Face geometry (face_r_dim, num_faces) and the current dst format (for the stride
 * baseline) are read from the operand's CB metadata.
 *
 * @param srcb_new_operand New operand id whose tile shape to program srcB for.
 */
inline void llk_unpack_reconfig_tile_shape_srcb(const std::uint32_t srcb_new_operand) {
    SAN_HOOK(unsupported());
    const std::uint32_t srcb_operand_id = get_operand_id(srcb_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srcb_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srcb_operand_id);
    // A tile-shape change is a tile-size change too; the LLK refreshes the tile-size GPR alongside geometry.
    assert_cb_page_size_matches_tile_size(srcb_operand_id, unpack_src_format[srcb_operand_id], face_r_dim, num_faces);
    _llk_unpack_reconfig_tile_shape_srcb_(unpack_src_format[srcb_operand_id], face_r_dim, num_faces);
}

/**
 * Conditionally reprogram the srcB unpacker tile/face geometry when switching operands. Reprograms only
 * when the CBs differ or the face geometry changed between the old and new operands.
 *
 * @param srcb_old_operand Currently configured srcB operand id.
 * @param srcb_new_operand New srcB operand id to switch to.
 */
inline void llk_unpack_reconfig_tile_shape_srcb(
    const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    SAN_HOOK(unsupported());
    const std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    const std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if (should_reconfigure_cbs(srcb_old_operand, srcb_new_operand) ||
        get_operand_face_r_dim(old_srcb_operand_id) != get_operand_face_r_dim(new_srcb_operand_id) ||
        get_operand_num_faces(old_srcb_operand_id) != get_operand_num_faces(new_srcb_operand_id)) {
        llk_unpack_reconfig_tile_shape_srcb(srcb_new_operand);
    }
}

/**
 * Mark srcB as holding dummy-valid data so the math thread can proceed without a real srcB unpack.
 */
inline void llk_unpack_set_srcb_dummy_valid() {
    SAN_HOOK(unsupported());
    _llk_unpack_set_srcb_dummy_valid_();
}
