// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_reduce.h"

/*************************************************************************
 * LLK REDUCE
 *************************************************************************/

/**
 *
 * @brief Initialize FPU to perform a reduce operation
 *
 * @tparam pool_type: Type of reduce pool op, values = [MAX, SUM, AVG]
 * @tparam reduce_dim: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @tparam math_fidelity: Only works for AVG/SUM pool types,  0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls
 * precision of multiplication
 * @param operand: The input operand circular buffer identifier
 *
 */
template <PoolType pool_type, ReduceDim reduce_dim, ckernel::MathFidelity math_fidelity>
inline void llk_math_reduce_init(const std::uint32_t operandA) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);
    _llk_math_reduce_init_<pool_type, reduce_dim, math_fidelity>(tensor_shape);
}

/**
 * @brief Perform a reduce operation
 *
 * @param dst_index: Tile index into the destination register.
 */
inline void llk_math_reduce(const std::uint32_t dst_index) { _llk_math_reduce_(dst_index); }
