// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../../../../../third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_custom_mm.h"
#include "llk_math_common_api.h"

template <bool transpose = false, bool split_acc = false>
inline void llk_math_custom_mm_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandB_id = get_operand_id(operandA);
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);

    _llk_math_custom_mm_init_<transpose, split_acc>(operandB_face_r_dim);
}

template <bool finalize = true>
inline void llk_math_custom_mm(const uint dst_index, const std::uint32_t kt_dim = 1) {
    _llk_math_custom_mm_<finalize>(dst_index, kt_dim);
}
