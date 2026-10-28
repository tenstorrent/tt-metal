// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK AB MATMUL
 *************************************************************************/

// Unified cores, shared by the CB-id API below and the LLKOperand API (experimental/2_0/). Matmul unpack is
// FORMAT-FREE at the op level (src/dst formats are programmed at compute_kernel_hw_startup<SrcOrder::Reverse>),
// so the cores take only the already-resolved geometry (face_r_dim / num_faces / partial_face per src) +
// runtime addresses + per-tile sizes. The role swap (in0 -> SrcB, in1 -> SrcA) is applied by the callers.
template <bool auto_ttsync = false, bool stream_narrow = false>
inline void llk_unpack_AB_matmul_init_impl(
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim,
    const std::uint32_t unpA_face_r_dim,
    const std::uint32_t unpB_face_r_dim,
    const std::uint32_t unpA_num_faces,
    const std::uint32_t unpB_num_faces,
    const bool partial_face_a,
    const bool partial_face_b) {
    _llk_unpack_AB_matmul_init_<0, 0, auto_ttsync, stream_narrow>(
        transpose,
        ct_dim,
        rt_dim,
        kt_dim,
        unpA_face_r_dim,
        unpB_face_r_dim,
        unpA_num_faces,
        unpB_num_faces,
        partial_face_a,
        partial_face_b);
}

template <bool auto_ttsync = false>
inline void llk_unpack_AB_matmul_impl(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const bool partial_face_a,
    const bool partial_face_b,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
    WAYPOINT("UPMW");
    _llk_unpack_AB_matmul_<
        0,
        0,
        auto_ttsync,
        get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>(),
        unpack_AB_matmul_hoist_rows<MATH_FIDELITY, DST_SYNC_MODE>()>(
        base_address_a,
        base_address_b,
        tile_index_a,
        tile_index_b,
        tile_size_a,
        tile_size_b,
        partial_face_a,
        partial_face_b,
        ct_dim,
        rt_dim,
        kt_dim);
    WAYPOINT("UPMD");
}

// Matmul in1 maps to hardware SrcA. CB page sizes already use unpacker address units.
inline void llk_unpack_AB_matmul_set_in1_column_stride(
    const std::uint32_t in1_cb_id, const std::uint32_t stride_tiles) {
    const std::uint32_t operand_id = get_operand_id(in1_cb_id);
    const std::uint32_t tile_size = get_local_cb_interface(operand_id).fifo_page_size;
    _llk_unpack_AB_matmul_set_in1_column_stride_(tile_size, stride_tiles);
}

// False when no unpack format of the kernel has 8 bits per datum or less: the 8-bit stream then folds away even in a
// kernel whose operand ids are run-time values.
constexpr bool llk_unpack_AB_matmul_kernel_has_narrow_format() {
    for (const std::uint8_t format : unpack_src_format) {
        if (_llk_unpack_AB_matmul_narrow_format_(format)) {
            return true;
        }
    }
    return false;
}

template <bool auto_ttsync = false>
__attribute__((always_inline)) inline void llk_unpack_AB_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0 -> srcB (supports partial face)
    // In1 -> srcA
    const std::uint32_t operandA_id = get_operand_id(operandB);
    const std::uint32_t operandB_id = get_operand_id(operandA);

    const std::uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const std::uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandB_id);

    const bool reuse_a = ct_dim >= rt_dim;
    constexpr std::uint32_t dest_max_tiles = get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>();
    constexpr bool may_stream_narrow = dest_max_tiles > 8 && llk_unpack_AB_matmul_kernel_has_narrow_format();
    const bool partial_face_a = get_operand_partial_face(operandA_id);
    const bool partial_face_b = get_operand_partial_face(operandB_id);

    const std::uint32_t unpA_num_faces = get_operand_num_faces(operandA_id);
    const std::uint32_t unpB_num_faces = get_operand_num_faces(operandB_id);  // if partial face -> unpack face by face

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        unpA_face_r_dim,
        unpB_face_r_dim,
        unpA_num_faces,
        unpB_num_faces));

    SAN_HOOK(init<OperationUnpackMatmul>(
        StateVal<OperationUnpackMatmul::Transpose>(transpose),
        StateVal<OperationUnpackMatmul::CtDim>(ct_dim),
        StateVal<OperationUnpackMatmul::RtDim>(rt_dim),
        StateVal<OperationUnpackMatmul::KtDim>(kt_dim),
        StateVal<OperationUnpackMatmul::PartialFaceA>(partial_face_a),
        StateVal<OperationUnpackMatmul::PartialFaceB>(partial_face_b),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::InputFormatB>(unpack_src_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatB>(unpack_dst_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightA>(unpA_face_r_dim),
        StateVal<Operand<Exu::Unpack>::FaceHeightB>(unpB_face_r_dim),
        StateVal<Operand<Exu::Unpack>::NumFacesA>(unpA_num_faces),
        StateVal<Operand<Exu::Unpack>::NumFacesB>(unpB_num_faces)));

    if (may_stream_narrow && _llk_unpack_AB_matmul_stream_narrow_(
                                 ct_dim, rt_dim, unpack_src_format[operandA_id], unpack_src_format[operandB_id], dest_max_tiles)) {
        llk_unpack_AB_matmul_init_impl<auto_ttsync, true>(
            transpose,
            ct_dim,
            rt_dim,
            kt_dim,
            unpA_face_r_dim,
            unpB_face_r_dim,
            unpA_num_faces,
            unpB_num_faces,
            partial_face_a,
            partial_face_b);
    } else {
        llk_unpack_AB_matmul_init_impl<auto_ttsync>(
            transpose,
            ct_dim,
            rt_dim,
            kt_dim,
            unpA_face_r_dim,
            unpB_face_r_dim,
            unpA_num_faces,
            unpB_num_faces,
            partial_face_a,
            partial_face_b);
    }
}

template <bool auto_ttsync = false>
inline void llk_unpack_AB_matmul(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0/InA -> srcB (supports partial face)
    // In1/InB -> srcA

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    // TODO: Review RT, use partial_face_b
    const bool partial_face_a = get_operand_partial_face(operandB_id);
    const bool partial_face_b = get_operand_partial_face(operandA_id);

    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;

    std::uint32_t tile_size_a = get_local_cb_interface(operandA_id).fifo_page_size;
    std::uint32_t tile_size_b = get_local_cb_interface(operandB_id).fifo_page_size;

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        get_operand_face_r_dim(operandB_id),
        get_operand_face_r_dim(operandA_id),
        get_operand_num_faces(operandB_id),
        get_operand_num_faces(operandA_id)));

    SAN_HOOK(execute<OperationUnpackMatmul>(
        StateVal<OperationUnpackMatmul::CtDim>(ct_dim),
        StateVal<OperationUnpackMatmul::RtDim>(rt_dim),
        StateVal<OperationUnpackMatmul::KtDim>(kt_dim),
        StateVal<OperationUnpackMatmul::PartialFaceA>(partial_face_a),
        StateVal<OperationUnpackMatmul::PartialFaceB>(partial_face_b),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::InputFormatB>(unpack_src_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatB>(unpack_dst_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightA>(get_operand_face_r_dim(operandB_id)),
        StateVal<Operand<Exu::Unpack>::FaceHeightB>(get_operand_face_r_dim(operandA_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesA>(get_operand_num_faces(operandB_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesB>(get_operand_num_faces(operandA_id)),
        StateDiscard<std::uint32_t>(tile_index_a),
        StateDiscard<std::uint32_t>(tile_index_b)));

    llk_unpack_AB_matmul_impl<auto_ttsync>(
        base_address_a,
        base_address_b,
        tile_index_a,
        tile_index_b,
        tile_size_a,
        tile_size_b,
        partial_face_a,
        partial_face_b,
        ct_dim,
        rt_dim,
        kt_dim);
}
