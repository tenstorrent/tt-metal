// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB
 *************************************************************************/

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const ckernel::Transpose transpose) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operandA_id],
        unpack_dst_format[operandA_id],
        unpack_src_format[get_operand_id(operandB)],
        unpack_dst_format[get_operand_id(operandB)],
        get_operand_face_r_dim(operandA_id),
        get_operand_face_r_dim(get_operand_id(operandB)),
        get_operand_num_faces(operandA_id),
        get_operand_num_faces(get_operand_id(operandB))));

    _llk_unpack_AB_init_<BType>(tensor_shape, transpose);
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
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0) {
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

    WAYPOINT("UABW");
    if constexpr (BType == BroadcastType::ROW) {
        _llk_unpack_AB_<BType>(address_a, address_b, bcast_row_idx, unpack_src_format[operandB_id]);
    } else {
        _llk_unpack_AB_<BType>(address_a, address_b);
    }
    WAYPOINT("UABD");
}
