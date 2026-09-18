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
// runtime addresses + the src format and tile shape the LLK sizes the per-tile L1 stride from. The role
// swap (in0 -> SrcB, in1 -> SrcA) is applied by the callers.
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
    _llk_unpack_AB_matmul_init_(
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

inline void llk_unpack_AB_matmul_impl(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t unpack_src_format_a,
    const std::uint32_t unpack_src_format_b,
    const ckernel::TensorShape tensor_shape_a,
    const ckernel::TensorShape tensor_shape_b,
    const bool partial_face_a,
    const bool partial_face_b,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
    WAYPOINT("UPMW");
    _llk_unpack_AB_matmul_(
        base_address_a,
        base_address_b,
        tile_index_a,
        tile_index_b,
        unpack_src_format_a,
        unpack_src_format_b,
        tensor_shape_a,
        tensor_shape_b,
        partial_face_a,
        partial_face_b,
        ct_dim,
        rt_dim,
        kt_dim);
    WAYPOINT("UPMD");
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

    const std::uint32_t unpA_face_r_dim = get_operand_face_r_dim(operandA_id);
    const std::uint32_t unpB_face_r_dim = get_operand_face_r_dim(operandB_id);

    const bool reuse_a = ct_dim >= rt_dim;
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

    llk_unpack_AB_matmul_init_impl(
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

    const ckernel::TensorShape tensor_shape_a = get_operand_tensor_shape(operandA_id);
    const ckernel::TensorShape tensor_shape_b = get_operand_tensor_shape(operandB_id);

    // The LLK derives the per-tile L1 stride from src format + tile shape. The CB page size the host
    // recorded must agree with it, otherwise tile-to-tile addressing walks the wrong stride.
    LLK_ASSERT(
        get_local_cb_interface(operandA_id).fifo_page_size ==
            _llk_unpack_tile_size_(
                unpack_src_format[operandA_id], tensor_shape_a.face_r_dim, tensor_shape_a.total_num_faces()),
        "operand A CB page size must equal the tile size derived from its src format and tile shape");
    LLK_ASSERT(
        get_local_cb_interface(operandB_id).fifo_page_size ==
            _llk_unpack_tile_size_(
                unpack_src_format[operandB_id], tensor_shape_b.face_r_dim, tensor_shape_b.total_num_faces()),
        "operand B CB page size must equal the tile size derived from its src format and tile shape");

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
        unpack_src_format[operandA_id],
        unpack_src_format[operandB_id],
        tensor_shape_a,
        tensor_shape_b,
        partial_face_a,
        partial_face_b,
        ct_dim,
        rt_dim,
        kt_dim);
}
