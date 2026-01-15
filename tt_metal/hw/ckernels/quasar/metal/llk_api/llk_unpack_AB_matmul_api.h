// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_matmul.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB MATMUL
 *************************************************************************/

/**
* @brief Initialize unpacker for matrix multiply

* @tparam TRANSPOSE_EN: Enables transpose of a tile
* @param operandA: The input0 operand circular buffer
* @param operandB: The input1 operand circular buffer
* @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
* @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
* @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
*
* This function initializes the unpacker to unpack operand 0 from the input0 operand circular buffer into SrcB
* and operand 1 from the input1 operand circular buffer into SrcA. Matrix multiply FPU operation does SrcB * SrcA.
*/
template <bool TRANSPOSE_EN = false>
__attribute__((always_inline)) inline void llk_unpack_AB_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0 -> srcB
    // In1 -> srcA
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    _llk_unpack_matmul_init_<TRANSPOSE_EN>(operandA_id, operandB_id, ct_dim, rt_dim, kt_dim);
}

/**
 *
 * @brief Performs unpack operation for matrix multiply such that:
 *
 * @param operandA: The input0 operand circular buffer
 * @param operandB: The input1 operand circular buffer
 * @param tile_index_a: The index into the input0 CB (UNPACKER1 -> SRCB)
 * @param tile_index_b: The index into the input1 CB (UNPACKER0 -> SRCA)
 * @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
 * @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
 * @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
 *
 * This function unpacks input0 and input1 operands from the input circular buffers to the src registers such that:
 * Input 0 -> unpack to SrcB
 * Input 1 -> unpack to SrcA
 * The matrix multiply has the following dimensions:
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim]
 * This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]
 * kt_dim is assumed to be iterated over outside this api call
 */
inline void llk_unpack_AB_matmul(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0/InA -> srcB
    // In1/InB -> srcA

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const std::uint32_t l1_tile_idx_0 = get_local_cb_interface(operandA_id).fifo_rd_tile_idx + tile_index_a;
    const std::uint32_t l1_tile_idx_1 = get_local_cb_interface(operandB_id).fifo_rd_tile_idx + tile_index_b;

    WAYPOINT("UPMW");
    _llk_unpack_matmul_(ct_dim, rt_dim, kt_dim, l1_tile_idx_0, l1_tile_idx_1);
    WAYPOINT("UPMD");
}
