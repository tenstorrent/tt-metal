// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_AB.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK AB
 *************************************************************************/

// Unified cores, shared by the CB-id API below and the LLKOperand API (experimental/2_0/). They take the
// already-resolved tile geometry / runtime addresses; the per-source prologue (resolving these from a CB id,
// or from an LLKMemDescriptor) lives in the callers. AB unpack is format-free at the op level: the src/dst
// formats are programmed once at compute_kernel_hw_startup, so the op needs only the two L1 addresses (plus
// the SrcB source format for the ROW-broadcast path).

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init_impl(const ckernel::TensorShape& tensor_shape, const ckernel::Transpose transpose) {
    _llk_unpack_AB_init_<BType>(tensor_shape, transpose);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_impl(
    const std::uint32_t address_a,
    const std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx,
    [[maybe_unused]] const std::uint32_t operandB_src_format) {
    WAYPOINT("UABW");
    if constexpr (BType == BroadcastType::ROW) {
        _llk_unpack_AB_<BType>(address_a, address_b, bcast_row_idx, operandB_src_format);
    } else {
        _llk_unpack_AB_<BType>(address_a, address_b);
    }
    WAYPOINT("UABD");
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const ckernel::Transpose transpose) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[get_operand_id(operandB)],
        unpack_dst_format[get_operand_id(operandB)],
        get_operand_face_r_dim(operandA_id),
        get_operand_face_r_dim(get_operand_id(operandB)),
        get_operand_num_faces(operandA_id),
        get_operand_num_faces(get_operand_id(operandB))));

    // _llk_unpack_AB_ takes address_a into SrcA and address_b into SrcB -- unlike the matmul
    // unpack, operandA/B are not swapped onto srcB/srcA here.
    SAN_HOOK(init<OperationUnpackBinary>(
        StateVal<OperationUnpackBinary::BroadcastType>(to_underlying(BType)),
        StateVal<OperationUnpackBinary::FaceWidth>(tensor_shape.face_c_dim),
        StateVal<OperationUnpackBinary::NumFacesRow>(tensor_shape.num_faces_r_dim),
        StateVal<OperationUnpackBinary::NumFacesCol>(tensor_shape.num_faces_c_dim),
        StateVal<OperationUnpackBinary::Transpose>(to_underlying(transpose)),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightA>(get_operand_face_r_dim(operandA_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesA>(get_operand_num_faces(operandA_id)),
        StateVal<Operand<Exu::Unpack>::InputFormatB>(unpack_src_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatB>(unpack_dst_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightB>(get_operand_face_r_dim(operandB_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesB>(get_operand_num_faces(operandB_id))));

    llk_unpack_AB_init_impl<BType>(tensor_shape, transpose);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    llk_unpack_AB_init<BType>(operandA, operandB, ckernel::Transpose::None);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t bcast_row_idx = 0) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_a = get_local_cb_interface(operandA_id).fifo_page_size * tile_index_a;
    std::uint32_t address_a = base_address_a + offset_address_a;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_b = get_local_cb_interface(operandB_id).fifo_page_size * tile_index_b;
    std::uint32_t address_b = base_address_b + offset_address_b;

    LLK_ASSERT(cb_access_within_bounds(operandA_id, tile_index_a, 1), "Indexed tile read exceeds CB boundary");
    LLK_ASSERT(cb_access_within_bounds(operandB_id, tile_index_b, 1), "Indexed tile read exceeds CB boundary");

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        get_operand_face_r_dim(operandA_id),
        get_operand_face_r_dim(operandB_id),
        get_operand_num_faces(operandA_id),
        get_operand_num_faces(operandB_id)));

    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    SAN_HOOK(execute<OperationUnpackBinary>(
        StateVal<OperationUnpackBinary::BroadcastType>(to_underlying(BType)),
        StateVal<OperationUnpackBinary::FaceWidth>(tensor_shape.face_c_dim),
        StateVal<OperationUnpackBinary::NumFacesRow>(tensor_shape.num_faces_r_dim),
        StateVal<OperationUnpackBinary::NumFacesCol>(tensor_shape.num_faces_c_dim),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightA>(get_operand_face_r_dim(operandA_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesA>(get_operand_num_faces(operandA_id)),
        StateVal<Operand<Exu::Unpack>::InputFormatB>(unpack_src_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatB>(unpack_dst_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightB>(get_operand_face_r_dim(operandB_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesB>(get_operand_num_faces(operandB_id)),
        StateDiscard<std::uint32_t>(tile_index_a),
        StateDiscard<std::uint32_t>(tile_index_b),
        StateDiscard<std::uint32_t>(bcast_row_idx)));

    llk_unpack_AB_impl<BType>(address_a, address_b, bcast_row_idx, unpack_src_format[operandB_id]);
}
