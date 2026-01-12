// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_reduce.h"

/*************************************************************************
 * LLK REDUCE
 *************************************************************************/

inline void llk_math_reduce(const uint dst_index) {
    _llk_math_reduce_(dst_index);
}

template <PoolType type, ReduceDim dim, int num_fidelity_phases = 0>
inline void llk_math_reduce_init(const std::uint32_t operandA) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    const TileShape tile_shape_A = {.num_faces = get_operand_num_faces(operand_id), .face_r_dim = get_operand_face_r_dim(operand_id), .face_c_dim = 16, .narrow_tile = get_operand_narrow_tile(operand_id)};
    _llk_math_reduce_init_<type, dim, num_fidelity_phases>(tile_shape_A);
}
