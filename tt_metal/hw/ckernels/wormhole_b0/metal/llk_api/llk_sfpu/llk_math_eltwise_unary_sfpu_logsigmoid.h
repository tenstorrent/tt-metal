// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_logsigmoid.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_logsigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::logsigmoid, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_logsigmoid(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_<APPROXIMATE>(
        ckernel::sfpu::calculate_logsigmoid<APPROXIMATE>, dst_index, vector_mode);
}

}  // namespace ckernel
