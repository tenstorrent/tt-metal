// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_matmul.h"

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

template <int NUM_FIDELITY_PHASES>
inline void llk_math_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    // TODO: this flags is only for computing 8x32 tile shape, although current impl assumes the in0 tile is still
    // 16x32. We should remove this flag in the furture and add impl for 8x32 input tile shape
    const bool partial_face = 0;

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    _llk_math_matmul_init_<NUM_FIDELITY_PHASES, DstTileFaceLayout::RowMajor>(
        in0_tile_r_dim,
        in0_tile_c_dim,
        in1_tile_r_dim,
        in1_tile_c_dim,
        partial_face,
        transpose,
        ct_dim,
        rt_dim,
        kt_dim);
}

template <int NUM_FIDELITY_PHASES, uint32_t num_faces = 4 /*not used*/>
inline void llk_math_matmul(
    const uint dst_index,
    const bool transpose = false,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    _llk_math_matmul_<NUM_FIDELITY_PHASES, DstTileFaceLayout::RowMajor>(dst_index, transpose, ct_dim, rt_dim, kt_dim);
}
