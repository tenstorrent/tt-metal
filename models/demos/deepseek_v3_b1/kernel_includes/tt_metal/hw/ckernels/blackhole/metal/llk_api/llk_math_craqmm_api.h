// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_craqmm.h"

/*************************************************************************
 * LLK CRAQMM (Custom Replay And Queue Matrix Multiply)
 *
 * Simplified implementation: THROTTLE_LEVEL = 0 (no throttling)
 *
 * Uses llk_math_craqmm.h as the low-level implementation.
 *************************************************************************/

template <int NUM_FIDELITY_PHASES>
inline void llk_math_craqmm_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t kt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    // Issue #31387: this flag is only for computing 8x32 tile shape, although current impl assumes the in0 tile is
    // still 16x32. We should remove this flag in the future and add impl for 8x32 input tile shape
    const bool partial_face = 0;

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    _llk_math_craqmm_init_<NUM_FIDELITY_PHASES>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, kt_dim);
}

template <int NUM_FIDELITY_PHASES, uint32_t num_faces = 4 /*not used*/>
inline void llk_math_craqmm(const uint dst_index, const bool transpose = false, const std::uint32_t kt_dim = 1) {
    _llk_math_craqmm_(dst_index, transpose, kt_dim);
}
