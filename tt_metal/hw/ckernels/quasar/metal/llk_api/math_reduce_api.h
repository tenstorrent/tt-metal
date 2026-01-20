// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
 * @tparam num_fidelity_phases: Only works for AVG/SUM pool types,  0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls
 * precision of multiplication
 * @param operand: The input operand circular buffer identifier
 *
 */
template <PoolType pool_type, ReduceDim reduce_dim, std::uint8_t num_fidelity_phases = 0>
inline void llk_math_reduce_init(const std::uint32_t operandA) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    const TileShape tile_shape_A = {
        .num_faces = get_operand_num_faces(operand_id),
        .face_r_dim = get_operand_face_r_dim(operand_id),
        .face_c_dim = FACE_C_DIM,
        .narrow_tile = get_operand_narrow_tile(operand_id)};

    _llk_math_reduce_init_<pool_type, reduce_dim, num_fidelity_phases>(tile_shape_A);
}

/**
 * @brief Perform a reduce operation
 *
 * @param dst_index: Tile index into the destination register.
 */
inline void llk_math_reduce(const uint dst_index) { _llk_math_reduce_(dst_index); }
