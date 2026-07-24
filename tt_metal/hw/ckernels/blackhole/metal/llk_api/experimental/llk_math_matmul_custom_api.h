// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "experimental/llk_math_matmul_custom_no_mop.h"

/*************************************************************************
 * LLK MATMUL NO MOP
 *************************************************************************/

// Operand tile geometry for the no-mop matmul, derived from each operand's ckernel::TensorShape
// (consistent with the other experimental SDPA ops, e.g. reduce_block_max_row / sub_bcast_col_custom,
// which obtain their geometry via get_operand_tensor_shape). A 16x32 tiny tile is num_faces_r_dim=1,
// num_faces_c_dim=2 (total 16x32); a full tile is 32x32.
struct MatmulNoMopGeom {
    std::uint32_t in0_tile_r_dim;
    std::uint32_t in0_tile_c_dim;
    std::uint32_t in1_tile_r_dim;
    std::uint32_t in1_tile_c_dim;
    bool partial_face;
};

inline MatmulNoMopGeom matmul_no_mop_geom(const std::uint32_t operandA, const std::uint32_t operandB) {
    const ckernel::TensorShape in0_shape = get_operand_tensor_shape(get_operand_id(operandA));
    const ckernel::TensorShape in1_shape = get_operand_tensor_shape(get_operand_id(operandB));
    const std::uint32_t in0_tile_r_dim = in0_shape.total_row_dim();
    const std::uint32_t in0_tile_c_dim = in0_shape.total_col_dim();
    const std::uint32_t in1_tile_r_dim = in1_shape.total_row_dim();
    const std::uint32_t in1_tile_c_dim = in1_shape.total_col_dim();
    return {in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, (in0_shape.face_r_dim < FACE_R_DIM)};
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init_no_mop(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const bool transpose = false,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    const MatmulNoMopGeom g = matmul_no_mop_geom(operandA, operandB);
    _llk_math_matmul_init_no_mop_<math_fidelity, THROTTLE_LEVEL>(
        g.in0_tile_r_dim, g.in0_tile_c_dim, g.in1_tile_r_dim, g.in1_tile_c_dim, g.partial_face, transpose, ct_dim, rt_dim);
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_no_mop(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const uint dst_index,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    // Re-derive operand tile geometry so the execute replays the geometry-correct length recorded at
    // init/reinit (16x32 tiny tiles record a shorter face-row-confined replay than full 32x32 tiles).
    const MatmulNoMopGeom g = matmul_no_mop_geom(operandA, operandB);
    _llk_math_matmul_no_mop_<math_fidelity, THROTTLE_LEVEL>(
        dst_index, ct_dim, rt_dim, g.in0_tile_r_dim, g.in0_tile_c_dim, g.in1_tile_r_dim, g.in1_tile_c_dim, g.partial_face);
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_reinit_no_mop(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const bool transpose = false,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    // Re-derive operand tile geometry so reinit restores the shape-aware addrmods for 16x32 tiny
    // tiles (SDPA reinits the QK^T matmul between q-subblocks). Mirrors llk_math_matmul_init_no_mop.
    const MatmulNoMopGeom g = matmul_no_mop_geom(operandA, operandB);
    matmul_configure_addrmod_reinit<math_fidelity, THROTTLE_LEVEL>(
        transpose, g.in0_tile_r_dim, g.in0_tile_c_dim, g.in1_tile_r_dim, g.in1_tile_c_dim, g.partial_face);
    math::reset_counters(p_setrwc::SET_ABD_F);
}
