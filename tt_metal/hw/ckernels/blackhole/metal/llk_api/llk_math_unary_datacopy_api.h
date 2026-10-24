// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

// Unified cores, shared by the CB-id API below and the LLKOperand API (experimental/). They take
// already-resolved scalar formats/geometry; the per-source prologue lives in the callers.
template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_impl(
    std::uint32_t dst_index,
    std::uint32_t src_format,
    std::uint32_t dst_format,
    const ckernel::TensorShape tensor_shape) {
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");
    _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
        dst_index, src_format, dst_format, tensor_shape);
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en = false,
    PackMode pack_mode = PackMode::Default>
inline void llk_math_eltwise_unary_datacopy_init_impl(
    const ckernel::TensorShape tensor_shape, std::uint32_t dst_format) {
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Tilize,
        "Blackhole math datacopy init supports only PackMode::Default and PackMode::Tilize");

    SAN_HOOK(init<OperationFpuEltwiseUnaryDatacopy>(
        StateVal<OperationFpuEltwiseUnaryDatacopy::DataCopyType>(to_underlying(type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::BroadcastType>(to_underlying(src_b_bcast_type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::NumFaces>(tensor_shape.total_num_faces()),
        StateVal<Operand<Exu::Fpu>::Format>(dst_format)));

    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en, pack_mode>(
        tensor_shape, dst_format);
}

// Kept for the experimental/2_0 callers; remove when they move to TensorShape.
// Main passed no geometry here and the LLK defaulted to 4 faces, which DEFAULT_TENSOR_SHAPE matches.
template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_impl(
    std::uint32_t dst_index, std::uint32_t src_format, std::uint32_t dst_format) {
    llk_math_eltwise_unary_datacopy_impl<type, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
        dst_index, src_format, dst_format, ckernel::DEFAULT_TENSOR_SHAPE);
}

// Kept for the experimental/2_0 callers; remove when they move to TensorShape.
// Outside of asserts the LLK only uses the total face count, so the 1x2 pick for num_faces == 2 matches main.
template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en = false,
    PackMode pack_mode = PackMode::Default>
inline void llk_math_eltwise_unary_datacopy_init_impl(std::uint32_t num_faces, std::uint32_t dst_format) {
    llk_math_eltwise_unary_datacopy_init_impl<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en, pack_mode>(
        ckernel::tensor_shape_from_num_faces(FACE_R_DIM, num_faces), dst_format);
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(std::uint32_t dst_index, std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    SAN_HOOK(execute<OperationFpuEltwiseUnaryDatacopy>(
        StateVal<OperationFpuEltwiseUnaryDatacopy::DataCopyType>(to_underlying(type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::BroadcastType>(to_underlying(src_b_bcast_type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::NumFaces>(get_operand_num_faces(operand_id)),
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[operand_id]),
        StateDiscard<std::uint32_t>(dst_index)));

    // The execute keeps main's 4-face Dest zero-flag clear on 32-bit unpack to dest (it never got the
    // operand's face count), so pass the default shape here, not the operand's.
    llk_math_eltwise_unary_datacopy_impl<type, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
        dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id], ckernel::DEFAULT_TENSOR_SHAPE);
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(
    std::uint32_t start_dst_index, std::uint32_t ntiles, std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    SAN_HOOK(execute<OperationFpuEltwiseUnaryDatacopy>(
        StateVal<OperationFpuEltwiseUnaryDatacopy::DataCopyType>(to_underlying(type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::BroadcastType>(to_underlying(src_b_bcast_type)),
        StateVal<OperationFpuEltwiseUnaryDatacopy::NumFaces>(get_operand_num_faces(operand_id)),
        StateVal<Operand<Exu::Fpu>::Format>(unpack_dst_format[operand_id]),
        StateDiscard<std::uint32_t>(start_dst_index),
        StateDiscard<std::uint32_t>(ntiles)));

    for (std::uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
        LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");

        // Same as llk_math_eltwise_unary_datacopy: keep main's 4-face Dest zero-flag clear.
        _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
            dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id], ckernel::DEFAULT_TENSOR_SHAPE);
    }
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en = false,
    PackMode pack_mode = PackMode::Default>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    llk_math_eltwise_unary_datacopy_init_impl<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en, pack_mode>(
        get_operand_tensor_shape(operand_id), get_operand_dst_format(operand_id));
}

template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {
    SAN_HOOK(uninit<OperationFpuEltwiseUnaryDatacopy>(
        StateVal<OperationFpuEltwiseUnaryDatacopy::BroadcastType>(to_underlying(src_b_bcast_type)),
        StateDiscard<bool>(unpack_to_dest)));
    _llk_math_eltwise_unary_datacopy_uninit_<src_b_bcast_type, unpack_to_dest>();
}
