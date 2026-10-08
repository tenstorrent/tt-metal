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

// src_dvalid must match the math init of the op (SrcDvalid in llk_defs.h).
template <BroadcastType BType = BroadcastType::NONE, SrcDvalid src_dvalid = SrcDvalid::PerFace>
inline void llk_unpack_AB_init_impl(const ckernel::TensorShape& tensor_shape, const ckernel::Transpose transpose) {
    _llk_unpack_AB_init_<BType, src_dvalid>(tensor_shape, transpose);
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

template <BroadcastType BType = BroadcastType::NONE, SrcDvalid src_dvalid = SrcDvalid::PerFace>
inline void llk_unpack_AB_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const ckernel::Transpose transpose) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        get_operand_face_r_dim(operandA_id),
        get_operand_face_r_dim(operandB_id),
        get_operand_num_faces(operandA_id),
        get_operand_num_faces(operandB_id)));

    SAN_HOOK(init<OperationUnpackBinary>(
        StateVal<OperationUnpackBinary::BroadcastType>(to_underlying(BType)),
        StateVal<OperationUnpackBinary::Transpose>(to_underlying(transpose)),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightA>(get_operand_face_r_dim(operandA_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesA>(get_operand_num_faces(operandA_id)),
        StateVal<Operand<Exu::Unpack>::InputFormatB>(unpack_src_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatB>(unpack_dst_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightB>(get_operand_face_r_dim(operandB_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesB>(get_operand_num_faces(operandB_id))));

    if constexpr (src_dvalid == SrcDvalid::PerTile && (BType == BroadcastType::COL || BType == BroadcastType::ROW)) {
        // Its SrcB layout reads B as 16-row faces: a row or column broadcast from a smaller B tile stays per face,
        // except a column broadcast of partial faces from a B tile of A's shape
        const std::uint32_t b_face_r_dim = get_operand_face_r_dim(operandB_id);
        const std::uint32_t b_num_faces = get_operand_num_faces(operandB_id);
        const bool b_like_a_partial_col = BType == BroadcastType::COL && tensor_shape.face_r_dim < FACE_R_DIM &&
                                          b_face_r_dim == tensor_shape.face_r_dim &&
                                          b_num_faces == tensor_shape.total_num_faces();
        if ((b_face_r_dim != FACE_R_DIM || b_num_faces != 4) && !b_like_a_partial_col) {
            llk_unpack_AB_init_impl<BType, SrcDvalid::PerFace>(tensor_shape, transpose);
            return;
        }
    }
    llk_unpack_AB_init_impl<BType, src_dvalid>(tensor_shape, transpose);
}

template <BroadcastType BType = BroadcastType::NONE, SrcDvalid src_dvalid = SrcDvalid::PerFace>
inline void llk_unpack_AB_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    llk_unpack_AB_init<BType, src_dvalid>(operandA, operandB, ckernel::Transpose::None);
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

    SAN_HOOK(execute<OperationUnpackBinary>(
        StateVal<OperationUnpackBinary::BroadcastType>(to_underlying(BType)),
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

// ntiles tile pairs from one config context: one context acquire per block instead of per tile. Tile i of A is
// start_tile_index_a + i * step_a, of B start_tile_index_b + i * step_b; a step of 0 reuses one tile.
inline void llk_unpack_AB_block(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t start_tile_index_a,
    const std::uint32_t start_tile_index_b,
    const std::uint32_t ntiles,
    const std::uint32_t step_a = 1,
    const std::uint32_t step_b = 1) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const std::uint32_t page_a = get_local_cb_interface(operandA_id).fifo_page_size;
    const std::uint32_t page_b = get_local_cb_interface(operandB_id).fifo_page_size;
    const std::uint32_t address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1 + page_a * start_tile_index_a;
    const std::uint32_t address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1 + page_b * start_tile_index_b;

    LLK_ASSERT(
        cb_access_within_bounds(operandA_id, start_tile_index_a, step_a * (ntiles - 1) + 1),
        "Block tile read exceeds CB boundary");
    LLK_ASSERT(
        cb_access_within_bounds(operandB_id, start_tile_index_b, step_b * (ntiles - 1) + 1),
        "Block tile read exceeds CB boundary");

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[operandB_id],
        unpack_dst_format[operandB_id],
        get_operand_face_r_dim(operandA_id),
        get_operand_face_r_dim(operandB_id),
        get_operand_num_faces(operandA_id),
        get_operand_num_faces(operandB_id)));

    // One execute per tile; the state is identical for every iteration, so it is restated once.
    SAN_HOOK(execute<OperationUnpackBinary>(
        StateVal<OperationUnpackBinary::BroadcastType>(to_underlying(BroadcastType::NONE)),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[operandA_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightA>(get_operand_face_r_dim(operandA_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesA>(get_operand_num_faces(operandA_id)),
        StateVal<Operand<Exu::Unpack>::InputFormatB>(unpack_src_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatB>(unpack_dst_format[operandB_id]),
        StateVal<Operand<Exu::Unpack>::FaceHeightB>(get_operand_face_r_dim(operandB_id)),
        StateVal<Operand<Exu::Unpack>::NumFacesB>(get_operand_num_faces(operandB_id)),
        StateDiscard<std::uint32_t>(start_tile_index_a),
        StateDiscard<std::uint32_t>(start_tile_index_b),
        StateDiscard<std::uint32_t>(ntiles)));

    WAYPOINT("UABW");
    _llk_unpack_AB_block_<BroadcastType::NONE>(address_a, address_b, ntiles, page_a * step_a, page_b * step_b);
    WAYPOINT("UABD");
}
