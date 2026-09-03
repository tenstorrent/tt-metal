// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_A.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

// Unified core, shared by the CB-id API below and the LLKOperand API (experimental/). It takes
// already-resolved scalar format/geometry + the runtime address; the per-source prologue (resolving
// these from a CB id, or from an MemDescriptor) lives in the callers.
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init_impl(
    const std::uint32_t transpose_of_faces,
    const std::uint32_t within_face_16x16_transpose,
    const ckernel::TensorShape tensor_shape,
    const std::uint32_t src_format,
    const std::uint32_t dst_format) {
    LLK_ASSERT_BLOCK((is_unpacker_A_configured_correctly<
                      UnpackerProgramType::ProgramByTile,
                      (BType != BroadcastType::NONE && !unpack_to_dest)>(
        src_format, dst_format, tensor_shape.face_r_dim, tensor_shape.total_num_faces())));

    _llk_unpack_A_init_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces, within_face_16x16_transpose, tensor_shape, src_format, dst_format);
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_impl(
    const std::uint32_t address, const std::uint32_t src_format, const std::uint32_t dst_format) {
    WAYPOINT("UPAW");
    _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(address, src_format, dst_format);
    WAYPOINT("UPAD");
}

// CB-id API (unchanged behavior): resolve format/geometry from the CB, then call the unified core.
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
    llk_unpack_A_init_impl<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces,
        within_face_16x16_transpose,
        get_operand_tensor_shape(operand_id),
        unpack_src_format[operand_id],
        unpack_dst_format[operand_id]);
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

    llk_unpack_A_impl<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        address, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
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
    _llk_unpack_A_uninit_<BType>();
}
