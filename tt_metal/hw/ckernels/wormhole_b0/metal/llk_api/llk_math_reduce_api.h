// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_reduce.h"

/*************************************************************************
 * LLK REDUCE
 *************************************************************************/

template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    int num_fidelity_phases = 0,
    bool is_int_fpu_en = false,
    bool enforce_fp32_accumulation = false>
inline void llk_math_reduce(const uint dst_index, const uint num_faces = 4) {
    _llk_math_reduce_<type, dim, is_fp32_dest_acc_en, num_fidelity_phases, is_int_fpu_en, enforce_fp32_accumulation>(
        dst_index, false, num_faces);
}

template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    int num_fidelity_phases = 0,
    bool is_int_fpu_en = false,
    bool enforce_fp32_accumulation = false>
inline void llk_math_reduce(const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t dst_index) {
    const std::uint32_t operand_id = get_operand_id(operandA);  // both operands must have same number of faces
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    _llk_math_reduce_<type, dim, is_fp32_dest_acc_en, num_fidelity_phases, is_int_fpu_en, enforce_fp32_accumulation>(
        dst_index, false, num_faces);
}

template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    int num_fidelity_phases = 0,
    bool enforce_fp32_accumulation = false>
inline void llk_math_reduce_init(
    const std::uint32_t within_face_16x16_transpose =
        0) {  // within_face_16x16_transpose used for unpack, ignored by math
    _llk_math_reduce_init_<type, dim, is_fp32_dest_acc_en, num_fidelity_phases, enforce_fp32_accumulation>(
        within_face_16x16_transpose);
}

// OPTIMIZED, DO NOT CALL UNLESS REGULAR TILE SIZE
/**
 * Initializes specialized reduce_max_row operation for single tile processing.
 *
 * NOTE: This function is highly specialized for SDPA (Scaled Dot-Product Attention) use cases
 * and should NOT be used as a substitute for the native llk_math_reduce_init LLK.
 * Use the standard llk_math_reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>() for general-purpose reduction.
 */
template <bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_max_row_init() {
    _llk_math_reduce_max_row_init_<is_fp32_dest_acc_en>();
}

// OPTIMIZED, DO NOT CALL UNLESS REGULAR TILE SIZE
/**
 * Performs specialized reduce_max_row operation on a single tile.
 *
 * NOTE: This function is highly specialized for SDPA (Scaled Dot-Product Attention) use cases
 * and should NOT be used as a substitute for the native llk_math_reduce LLK.
 * Use the standard llk_math_reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>() for general-purpose reduction.
 */
inline void llk_math_reduce_max_row(const uint dst_index) { _llk_math_reduce_max_row_(dst_index); }

// Block-based reduce row max functions
/**
 * Initializes block-based reduce_max_row operation for processing multiple tiles.
 *
 * NOTE: This function is highly specialized for SDPA (Scaled Dot-Product Attention) use cases
 * and should NOT be used as a substitute for the native llk_math_reduce_init LLK.
 * Use the standard llk_math_reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>() with multiple
 * llk_math_reduce() calls in a loop for general-purpose block reduction.
 */
template <uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row_init() {
    _llk_math_reduce_block_max_row_init_<block_ct_dim, is_fp32_dest_acc_en>();
}

/**
 * Performs block-based reduce_max_row operation across multiple tiles in the width dimension.
 *
 * NOTE: This function is highly specialized for SDPA (Scaled Dot-Product Attention) use cases
 * and should NOT be used as a substitute for the native llk_math_reduce LLK.
 * Use the standard llk_math_reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>() in a loop
 * for general-purpose block reduction across multiple tiles.
 */
template <uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void llk_math_reduce_block_max_row(const uint dst_index) {
    _llk_math_reduce_block_max_row_<block_ct_dim, is_fp32_dest_acc_en>(dst_index);
}
