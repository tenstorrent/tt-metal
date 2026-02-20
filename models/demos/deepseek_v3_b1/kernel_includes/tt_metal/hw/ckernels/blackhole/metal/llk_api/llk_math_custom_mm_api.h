// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../../../../../third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_custom_mm.h"
#include "llk_math_common_api.h"

/*************************************************************************
 * LLK MATH CUSTOM_MM
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: {1, 2, 4, 6, 8, 10, 12, 14, 16}
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Uses llk_math_custom_mm.h as the low-level implementation.
 *************************************************************************/

template <bool transpose = false, bool split_acc = false, bool dense_packing = false>
inline void llk_math_custom_mm_init(
    const std::uint32_t operand0, const std::uint32_t operand1, const std::uint32_t ct_dim = 1) {
    // Swap operands, for matmul operand0 goes to SrcB and operand1 goes to SrcA
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_math_custom_mm_init_<transpose, split_acc, dense_packing>(operandB_face_r_dim, ct_dim);
}

template <bool finalize = true>
inline void llk_math_custom_mm(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    // Swap operands, for matmul operand0 goes to SrcB and operand1 goes to SrcA
    const std::uint32_t operandB_id = get_operand_id(operand0);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_math_custom_mm_<finalize>(operandB_face_r_dim, dst_index, kt_dim, ct_dim);
}
