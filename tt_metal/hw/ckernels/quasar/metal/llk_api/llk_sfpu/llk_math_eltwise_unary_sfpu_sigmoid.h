// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_assert.h"

namespace ckernel {

template <[[maybe_unused]] bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid>();
}

template <
    [[maybe_unused]] bool APPROXIMATE,
    [[maybe_unused]] bool is_fp32_dest_acc_en,
    int ITERATIONS = SFPU_ITERATIONS>
inline void llk_math_eltwise_unary_sfpu_sigmoid(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    LLK_ASSERT(vector_mode == (int)VectorMode::RC, "Quasar currently only supports vector mode RC");
    _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_sigmoid, dst_index, ITERATIONS);
}

}  // namespace ckernel
