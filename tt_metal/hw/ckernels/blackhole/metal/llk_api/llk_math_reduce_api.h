// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
    MathFidelity math_fidelity,
    bool is_int_fpu_en = false,
    bool enforce_fp32_accumulation = false,
    bool tilize_AB_support_en = true>
inline void llk_math_reduce(const uint dst_index, const uint num_faces = 4) {
    _llk_math_reduce_<type, dim, is_fp32_dest_acc_en, math_fidelity, is_int_fpu_en, enforce_fp32_accumulation>(
        dst_index, false, num_faces);
}

template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool is_int_fpu_en = false,
    bool enforce_fp32_accumulation = false,
    bool tilize_AB_support_en = true>
inline void llk_math_reduce(const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t dst_index) {
    const std::uint32_t operand_id = get_operand_id(operandA);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_reduce_<type, dim, is_fp32_dest_acc_en, math_fidelity, is_int_fpu_en, enforce_fp32_accumulation>(
        dst_index, false, num_faces);
}

template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool enforce_fp32_accumulation = false /*unused*/>
inline void llk_math_reduce_init() {
    _llk_math_reduce_init_<type, dim, is_fp32_dest_acc_en, math_fidelity, enforce_fp32_accumulation>();
}

template <bool enforce_fp32_accumulation = false>
inline void llk_math_reduce_uninit() {
    _llk_math_reduce_uninit_<enforce_fp32_accumulation>();
}
