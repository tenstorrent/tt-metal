// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_matmul.h"

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

template <int NUM_FIDELITY_PHASES, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    // Determine if we have partial faces (tiny tiles)
    // The matmul operation is D = B*A where in0->srcB and in1->srcA
    // A partial face occurs when tile has non-standard face dimensions (row or col dim != 16)
    // For matmul address mode configuration, we specifically need to check if in0 (srcB) has partial faces
    // Example: 8x32 tile has 8<=16 rows and 32>16 cols, requiring special address mode handling
    const bool is_in0_16x32 = (in0_tile_r_dim <= 16) && (in0_tile_c_dim > 16);
    const bool is_in0_32x16 = (in0_tile_r_dim > 16) && (in0_tile_c_dim <= 16);
    // A tile is partial face if it's not the standard 32x32 AND has mismatched dimensions (one dim <=16, other >16)
    const bool partial_face = (is_in0_16x32 || is_in0_32x16);

    _llk_math_matmul_init_<NUM_FIDELITY_PHASES, THROTTLE_LEVEL>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}

template <int NUM_FIDELITY_PHASES, int THROTTLE_LEVEL = 0, uint32_t num_faces = 4 /*not used*/>
inline void llk_math_matmul(const uint dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    _llk_math_matmul_<NUM_FIDELITY_PHASES, THROTTLE_LEVEL>(dst_index, ct_dim, rt_dim);
}
