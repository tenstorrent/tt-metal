// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_sigmoid_appx.h"

namespace ckernel {

inline void llk_math_eltwise_unary_sfpu_sigmoid_appx_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid_appx>(sfpu::sigmoid_appx_init);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sigmoid_appx(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL_FN(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sigmoid_appx, dst_index, vector_mode);
}

}  // namespace ckernel
