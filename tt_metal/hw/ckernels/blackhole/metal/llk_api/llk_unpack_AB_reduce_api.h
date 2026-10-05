// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK AB REDUCE
 *************************************************************************/

// Unified cores, shared by the CB-id API below and the LLKOperand API (experimental/2_0/). Reduce unpack is
// FORMAT-FREE at the op level (formats set at compute_kernel_hw_startup), so the cores take only operand A's
// tile geometry (init) / the two runtime L1 addresses (exec). Callers resolve these from a CB id or a
// descriptor.
template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce_init_impl(const ckernel::TensorShape& tensor_shape) {
    SAN_HOOK(init<OperationUnpackReduce>(
        StateVal<OperationUnpackReduce::PoolType>(to_underlying(pool_type)),
        StateVal<OperationUnpackReduce::ReduceDim>(to_underlying(reduce_dim)),
        StateVal<OperationUnpackReduce::FaceHeight>(tensor_shape.face_r_dim),
        StateVal<OperationUnpackReduce::NumFaces>(tensor_shape.total_num_faces())));

    _llk_unpack_AB_reduce_init_<pool_type, reduce_dim>(tensor_shape);
}

template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce_impl(const std::uint32_t address_a, const std::uint32_t address_b) {
    WAYPOINT("UABW");
    _llk_unpack_AB_reduce_<pool_type, reduce_dim>(address_a, address_b);
    WAYPOINT("UABD");
}

template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    llk_unpack_AB_reduce_init_impl<pool_type, reduce_dim>(tensor_shape);
}

template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b) {
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

    // SUM/AVG REDUCE_ROW swaps the operands (scaler -> SrcA, data -> SrcB), see _llk_unpack_AB_reduce_.
    [[maybe_unused]] constexpr bool swap_operands =
        (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX);
    [[maybe_unused]] const std::uint32_t unpA_operand_id = swap_operands ? operandB_id : operandA_id;
    [[maybe_unused]] const std::uint32_t unpB_operand_id = swap_operands ? operandA_id : operandB_id;

    SAN_HOOK(execute<OperationUnpackReduce>(
        StateVal<OperationUnpackReduce::PoolType>(to_underlying(pool_type)),
        StateVal<OperationUnpackReduce::ReduceDim>(to_underlying(reduce_dim)),
        StateVal<OperationUnpackReduce::FaceHeight>(get_operand_face_r_dim(operandA_id)),
        StateVal<OperationUnpackReduce::NumFaces>(get_operand_num_faces(operandA_id)),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[unpA_operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[unpA_operand_id]),
        StateVal<Operand<Exu::Unpack>::InputFormatB>(unpack_src_format[unpB_operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatB>(unpack_dst_format[unpB_operand_id]),
        StateDiscard<std::uint32_t>(tile_index_a),
        StateDiscard<std::uint32_t>(tile_index_b)));

    llk_unpack_AB_reduce_impl<pool_type, reduce_dim>(address_a, address_b);
}

// Block of num_tiles consecutive tiles of operandA from tile_index_a, against the scaler tile tile_index_b of operandB.
template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce_block(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t num_tiles) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t page_size_a = get_local_cb_interface(operandA_id).fifo_page_size;
    std::uint32_t address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1 + page_size_a * tile_index_a;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_b = get_local_cb_interface(operandB_id).fifo_page_size * tile_index_b;
    std::uint32_t address_b = base_address_b + offset_address_b;

    LLK_ASSERT(cb_access_within_bounds(operandA_id, tile_index_a, num_tiles), "Block tile read exceeds CB boundary");
    LLK_ASSERT(cb_access_within_bounds(operandB_id, tile_index_b, 1), "Indexed tile read exceeds CB boundary");

    [[maybe_unused]] constexpr bool swap_operands =
        (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX);
    [[maybe_unused]] const std::uint32_t unpA_operand_id = swap_operands ? operandB_id : operandA_id;
    [[maybe_unused]] const std::uint32_t unpB_operand_id = swap_operands ? operandA_id : operandB_id;

    // One execute per tile; the state is identical for every tile, so it is restated once.
    SAN_HOOK(execute<OperationUnpackReduce>(
        StateVal<OperationUnpackReduce::PoolType>(to_underlying(pool_type)),
        StateVal<OperationUnpackReduce::ReduceDim>(to_underlying(reduce_dim)),
        StateVal<OperationUnpackReduce::FaceHeight>(get_operand_face_r_dim(operandA_id)),
        StateVal<OperationUnpackReduce::NumFaces>(get_operand_num_faces(operandA_id)),
        StateVal<Operand<Exu::Unpack>::InputFormatA>(unpack_src_format[unpA_operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatA>(unpack_dst_format[unpA_operand_id]),
        StateVal<Operand<Exu::Unpack>::InputFormatB>(unpack_src_format[unpB_operand_id]),
        StateVal<Operand<Exu::Unpack>::OutputFormatB>(unpack_dst_format[unpB_operand_id]),
        StateDiscard<std::uint32_t>(tile_index_a),
        StateDiscard<std::uint32_t>(tile_index_b)));

    WAYPOINT("UABW");
    _llk_unpack_AB_reduce_block_<pool_type, reduce_dim>(
        address_a,
        address_b,
        num_tiles,
        page_size_a,
        unpack_src_format[operandA_id],
        get_operand_tensor_shape(operandA_id));
    WAYPOINT("UABD");
}
