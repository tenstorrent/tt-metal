// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "experimental/llk_math_matmul_custom_no_mop.h"

/*************************************************************************
 * LLK MATMUL NO MOP
 *************************************************************************/

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init_no_mop(
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

    const bool partial_face = (in0_tile_r_dim < FACE_R_DIM);

    _llk_math_matmul_init_no_mop_<static_cast<int>(math_fidelity), THROTTLE_LEVEL>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_no_mop(
    const uint dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    _llk_math_matmul_no_mop_<static_cast<int>(math_fidelity), THROTTLE_LEVEL>(dst_index, ct_dim, rt_dim);
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_reinit_no_mop(
    const std::uint32_t transpose = 0, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    (void)math_fidelity;
    (void)THROTTLE_LEVEL;
    (void)ct_dim;
    (void)rt_dim;
    matmul_configure_addrmod_reinit<static_cast<int>(math_fidelity), THROTTLE_LEVEL>(transpose);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_configure_addrmod_reinit(const std::uint32_t transpose = 0) {
    matmul_configure_addrmod_reinit<static_cast<int>(math_fidelity), THROTTLE_LEVEL>(transpose);
}
