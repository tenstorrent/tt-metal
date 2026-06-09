// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB
 *************************************************************************/

/**
 * @brief Initialize unpacker to unpack two source operands A and B into SrcA and SrcB registers.
 *
 * Derives the tile shape from operand A's circular buffer and configures the unpacker hardware for
 * dual-operand unpacking with the requested broadcast mode and optional transpose.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param operandA: Circular-buffer index of source A.
 * @param operandB: Circular-buffer index of source B.
 * @param transpose: Transpose mode for SrcA face order and/or within-face transpose, values =
 * <None/IntraFace/InterFace/Both>
 * @ref llk_unpack_AB is the matching execute call.
 * @ref llk_math_eltwise_binary_init is the matching init on the math thread (consumes SrcA/SrcB).
 */
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

/**
 * @brief Initialize unpacker to unpack operands A and B with no transpose.
 *
 * Convenience overload that forwards to the transpose-aware init with @ref ckernel::Transpose::None.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param operandA: Circular-buffer index of source A.
 * @param operandB: Circular-buffer index of source B.
 * @ref llk_unpack_AB is the matching execute call.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    llk_unpack_AB_init<BType>(operandA, operandB, ckernel::Transpose::None);
}

/**
 * @brief Unpack two tiles from L1 memory into SrcA and SrcB registers.
 *
 * Resolves each tile's L1 address from its operand's circular buffer and tile index, then runs the
 * configured MOP.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param operandA: Circular-buffer index of source A.
 * @param operandB: Circular-buffer index of source B.
 * @param tile_index_a: Index of the source A tile within its circular buffer.
 * @param tile_index_b: Index of the source B tile within its circular buffer.
 * @param bcast_row_idx: Row index within the source B tile for ROW broadcast.
 * @note Call @ref llk_unpack_AB_init with matching template args before this function.
 * @ref llk_math_eltwise_binary on the math thread consumes the SrcA/SrcB tiles unpacked here.
 */
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
