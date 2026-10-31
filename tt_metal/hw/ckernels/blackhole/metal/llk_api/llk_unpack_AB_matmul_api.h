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
// so the cores take only the in0/in1 tile shapes + runtime addresses + per-tile sizes. The LLK maps
// in0 -> SrcB and in1 -> SrcA.
inline void llk_unpack_AB_matmul_init_impl(
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim,
    const ckernel::TensorShape in0_tensor_shape,
    const ckernel::TensorShape in1_tensor_shape) {
    _llk_unpack_AB_matmul_init_(transpose, ct_dim, rt_dim, kt_dim, in0_tensor_shape, in1_tensor_shape);
}

inline void llk_unpack_AB_matmul_impl(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const ckernel::TensorShape in0_tensor_shape,
    const ckernel::TensorShape in1_tensor_shape,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
    WAYPOINT("UPMW");
    _llk_unpack_AB_matmul_(
        base_address_a,
        base_address_b,
        tile_index_a,
        tile_index_b,
        tile_size_a,
        tile_size_b,
        in0_tensor_shape,
        in1_tensor_shape,
        ct_dim,
        rt_dim,
        kt_dim);
    WAYPOINT("UPMD");
}

// Kept for the experimental/2_0 callers; remove when they move to TensorShape.
// Per source, the LLK reads only face_r_dim, total_num_faces() and total_row_dim() < TILE_R_DIM (unpack face by
// face). The face grid below gives back all three exactly: 4 faces -> 2x2, 1 face -> 1x1, and 2 faces -> 1x2
// when partial (16x32, 8x32, ...) or 2x1 when not (32x16). unpA (SrcA) is in1, unpB (SrcB) is in0.
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
    const auto shape_from_legacy =
        [](const std::uint32_t face_r_dim, const std::uint32_t num_faces, const bool partial_face) {
            const std::uint8_t num_faces_r_dim = (num_faces == 4 || (num_faces == 2 && !partial_face)) ? 2 : 1;
            const std::uint8_t num_faces_c_dim = (num_faces == 4 || (num_faces == 2 && partial_face)) ? 2 : 1;
            return ckernel::make_tensor_shape(
                static_cast<std::uint8_t>(face_r_dim),
                static_cast<std::uint8_t>(FACE_C_DIM),
                num_faces_r_dim,
                num_faces_c_dim);
        };
    const ckernel::TensorShape in0_tensor_shape = shape_from_legacy(unpB_face_r_dim, unpB_num_faces, partial_face_b);
    const ckernel::TensorShape in1_tensor_shape = shape_from_legacy(unpA_face_r_dim, unpA_num_faces, partial_face_a);
    LLK_ASSERT(
        (in1_tensor_shape.total_row_dim() < TILE_R_DIM) == partial_face_a &&
            (in0_tensor_shape.total_row_dim() < TILE_R_DIM) == partial_face_b,
        "matmul partial_face does not match face_r_dim / num_faces");
    llk_unpack_AB_matmul_init_impl(transpose, ct_dim, rt_dim, kt_dim, in0_tensor_shape, in1_tensor_shape);
}

// Kept for the experimental/2_0 callers; remove when they move to TensorShape.
// The execute reads only total_row_dim() < TILE_R_DIM from each shape, so 16x32 (partial) or 32x32 (not partial)
// gives back the passed flag exactly. partial_face_a is SrcA (in1), partial_face_b is SrcB (in0), as on main.
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
    constexpr ckernel::TensorShape partial_shape = ckernel::make_tensor_shape(FACE_R_DIM, FACE_C_DIM, 1, 2);  // 16x32
    llk_unpack_AB_matmul_impl(
        base_address_a,
        base_address_b,
        tile_index_a,
        tile_index_b,
        tile_size_a,
        tile_size_b,
        partial_face_b ? partial_shape : ckernel::DEFAULT_TENSOR_SHAPE,  // in0 -> SrcB
        partial_face_a ? partial_shape : ckernel::DEFAULT_TENSOR_SHAPE,  // in1 -> SrcA
        ct_dim,
        rt_dim,
        kt_dim);
}

// Matmul in1 maps to hardware SrcA. CB page sizes already use unpacker address units.
inline void llk_unpack_AB_matmul_set_in1_column_stride(
    const std::uint32_t in1_cb_id, const std::uint32_t stride_tiles) {
    const std::uint32_t operand_id = get_operand_id(in1_cb_id);
    const std::uint32_t tile_size = get_local_cb_interface(operand_id).fifo_page_size;
    _llk_unpack_AB_matmul_set_in1_column_stride_(tile_size, stride_tiles);
}

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

    [[maybe_unused]] const std::uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    [[maybe_unused]] const std::uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandB_id);

    const bool reuse_a = ct_dim >= rt_dim;
    [[maybe_unused]] const bool partial_face_a = get_operand_partial_face(operandA_id);
    [[maybe_unused]] const bool partial_face_b = get_operand_partial_face(operandB_id);

    [[maybe_unused]] const std::uint32_t unpA_num_faces = get_operand_num_faces(operandA_id);
    [[maybe_unused]] const std::uint32_t unpB_num_faces =
        get_operand_num_faces(operandB_id);  // if partial face -> unpack face by face

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

    llk_unpack_AB_matmul_init_impl(
        transpose,
        ct_dim,
        rt_dim,
        kt_dim,
        get_operand_tensor_shape(operandB_id),   // in0 -> SrcB
        get_operand_tensor_shape(operandA_id));  // in1 -> SrcA
}

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

    // partial_face is only recorded for the sanitizer; the LLK derives it from the TensorShapes.
    [[maybe_unused]] const bool partial_face_a = get_operand_partial_face(operandB_id);
    [[maybe_unused]] const bool partial_face_b = get_operand_partial_face(operandA_id);

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

    llk_unpack_AB_matmul_impl(
        base_address_a,
        base_address_b,
        tile_index_a,
        tile_index_b,
        tile_size_a,
        tile_size_b,
        get_operand_tensor_shape(operandA_id),  // in0 -> SrcB
        get_operand_tensor_shape(operandB_id),  // in1 -> SrcA
        ct_dim,
        rt_dim,
        kt_dim);
}
