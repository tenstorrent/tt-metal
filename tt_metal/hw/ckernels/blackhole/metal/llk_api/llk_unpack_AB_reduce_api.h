// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB REDUCE
 *************************************************************************/

// Unified cores, shared by the CB-id API below and the LLKOperand API (experimental/2_0/). Reduce unpack is
// FORMAT-FREE at the op level (formats set at compute_kernel_hw_startup), so the cores take only operand A's
// tile geometry (init) / the two runtime L1 addresses (exec). Callers resolve these from a CB id or a
// descriptor.
template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce_init_impl(const ckernel::TensorShape& tensor_shape) {
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

    llk_unpack_AB_reduce_impl<pool_type, reduce_dim>(address_a, address_b);
}
