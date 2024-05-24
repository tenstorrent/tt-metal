// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_elu.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_elu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::elu, APPROXIMATE>(sfpu::elu_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_elu(uint dst_index, uint param0) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE>(
        ckernel::sfpu::calculate_elu<APPROXIMATE>,
        ckernel::sfpu::calculate_elu<APPROXIMATE>,
        dst_index,
        (int)VectorMode::RC,
        param0);
}

}  // namespace ckernel
