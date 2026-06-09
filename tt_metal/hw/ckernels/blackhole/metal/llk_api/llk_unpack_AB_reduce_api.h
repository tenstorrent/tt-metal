// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB REDUCE
 *************************************************************************/

/**
 * @brief Initialize the unpacker for reduce operations.
 *
 * Configures the unpacker hardware registers and MOP for reduction, including haloize mode for
 * row reductions and the unpacker X-dimension endpoints. Derives the tile geometry from
 * operand A's circular buffer.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam enforce_fp32_accumulation: Configure the ALU for FP32 accumulation (MOVB2D hi16/lo16 path).
 * @param operandA: Circular-buffer index of source operand A.
 * @param operandB: Circular-buffer index of the scaler operand B.
 * @note For REDUCE_ROW operations, the face is transposed using haloize mode.
 * @note Call @ref llk_unpack_AB_reduce with matching template args after this.
 * @ref llk_math_reduce_init is the matching init on the math thread (this is the scaler-operand unpack pairing).
 */
template <PoolType pool_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
inline void llk_unpack_AB_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);

    if constexpr (enforce_fp32_accumulation) {
        // Set necessary config regs for MOVB2D hi16/lo16 to work
        _llk_unpack_dbg_feature_disable_();
    }

    _llk_unpack_AB_reduce_init_<pool_type, reduce_dim, enforce_fp32_accumulation>(tensor_shape);
}

/**
 * @brief Execute the unpacker for reduction operations.
 *
 * Resolves each tile's L1 address from its operand's circular buffer, programs the source A and
 * B base addresses, synchronizes with Trisc via semaphores, and runs the configured MOP.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param operandA: Circular-buffer index of source operand A.
 * @param operandB: Circular-buffer index of the scaler operand B.
 * @param tile_index_a: Index of the tile to unpack within operand A's buffer.
 * @param tile_index_b: Index of the tile to unpack within operand B's buffer.
 * @note Call @ref llk_unpack_AB_reduce_init with matching template args before this function.
 */
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

    WAYPOINT("UABW");
    _llk_unpack_AB_reduce_<pool_type, reduce_dim>(address_a, address_b);
    WAYPOINT("UABD");
}
