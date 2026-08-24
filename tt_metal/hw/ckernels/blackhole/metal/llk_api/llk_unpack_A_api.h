// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_A.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    const std::uint32_t transpose_of_faces,
    const std::uint32_t within_face_16x16_transpose,
    const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    const std::uint32_t operand_unpack_src_format = unpack_src_format[operand_id];
    const std::uint32_t operand_unpack_dst_format = unpack_dst_format[operand_id];

    LLK_ASSERT_BLOCK((is_unpacker_A_configured_correctly<
                      UnpackerProgramType::ProgramByTile,
                      (BType != BroadcastType::NONE && !unpack_to_dest)>(
        operand_unpack_src_format,
        operand_unpack_dst_format,
        tensor_shape.face_r_dim,
        tensor_shape.total_num_faces())));

    // transpose_of_faces and within_face_16x16_transpose are not OperationUnpackUnary fields --
    // transpose is modelled on the matmul operation, not the unary one.
    SAN_HOOK(init<OperationUnpackUnary>(
        StateVal<OperationUnpackUnary::BroadcastType>(to_underlying(BType)),
        StateVal<OperationUnpackUnary::AccumulateToDest>(acc_to_dest),
        StateVal<OperationUnpackUnary::BinaryReuseDest>(to_underlying(binary_reuse_dest)),
        StateVal<OperationUnpackUnary::UnpackToDest>(unpack_to_dest),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(operand_unpack_src_format),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(operand_unpack_dst_format),
        StateVal<Operand<Exu::Unpack>::FaceHeightA>(tensor_shape.face_r_dim),
        StateVal<Operand<Exu::Unpack>::NumFacesA>(tensor_shape.total_num_faces()),
        StateDiscard<std::uint32_t>(transpose_of_faces),
        StateDiscard<std::uint32_t>(within_face_16x16_transpose)));

    _llk_unpack_A_init_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces,
        within_face_16x16_transpose,
        tensor_shape,
        operand_unpack_src_format,
        operand_unpack_dst_format);
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    std::uint32_t address = base_address + offset_address;

    LLK_ASSERT(cb_access_within_bounds(operand_id, tile_index, 1), "Indexed tile read exceeds CB boundary");

    LLK_ASSERT_BLOCK((is_unpacker_A_configured_correctly<
                      UnpackerProgramType::ProgramByTile,
                      (BType != BroadcastType::NONE && !unpack_to_dest)>(
        unpack_src_format[operand_id],
        unpack_dst_format[operand_id],
        get_operand_face_r_dim(operand_id),
        get_operand_num_faces(operand_id))));

    SAN_HOOK(execute<OperationUnpackUnary>(
        StateVal<OperationUnpackUnary::BroadcastType>(to_underlying(BType)),
        StateVal<OperationUnpackUnary::AccumulateToDest>(acc_to_dest),
        StateVal<OperationUnpackUnary::BinaryReuseDest>(to_underlying(binary_reuse_dest)),
        StateVal<OperationUnpackUnary::UnpackToDest>(unpack_to_dest),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[operand_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightA>(get_operand_face_r_dim(operand_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesA>(get_operand_num_faces(operand_id)),
        StateDiscard<std::uint32_t>(tile_index)));

    WAYPOINT("UPAW");
    _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        address, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
    WAYPOINT("UPAD");
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_block(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size;
    std::uint32_t address = base_address + start_tile_index * offset_address;

    LLK_ASSERT(cb_access_within_bounds(operand_id, start_tile_index, ntiles), "Block tile read exceeds CB boundary");

    // One execute per tile; the state is identical for every iteration, so it is restated once.
    SAN_HOOK(execute<OperationUnpackUnary>(
        StateVal<OperationUnpackUnary::BroadcastType>(to_underlying(BType)),
        StateVal<OperationUnpackUnary::AccumulateToDest>(acc_to_dest),
        StateVal<OperationUnpackUnary::BinaryReuseDest>(to_underlying(binary_reuse_dest)),
        StateVal<OperationUnpackUnary::UnpackToDest>(unpack_to_dest),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[operand_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightA>(get_operand_face_r_dim(operand_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesA>(get_operand_num_faces(operand_id)),
        StateDiscard<std::uint32_t>(start_tile_index),
        StateDiscard<std::uint32_t>(ntiles)));

    for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
            address, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
        address += offset_address;
        WAYPOINT("UPAD");
    }
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_A_uninit() {
    // No operand is in scope here; restating the broadcast variant is what ties the uninit to
    // the operation that was initialised.
    SAN_HOOK(uninit<OperationUnpackUnary>(StateVal<OperationUnpackUnary::BroadcastType>(to_underlying(BType))));

    _llk_unpack_A_uninit_<BType>();
}
