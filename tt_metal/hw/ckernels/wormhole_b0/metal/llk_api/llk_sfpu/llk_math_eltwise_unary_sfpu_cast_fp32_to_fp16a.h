// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_cast_fp32_to_fp16a.h"

namespace ckernel {

inline void llk_math_eltwise_unary_sfpu_cast_fp32_to_fp16a_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::cast_fp32_to_fp16a>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cast_fp32_to_fp16a(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_cast_fp32_to_fp16a, (APPROXIMATE), dst_index, vector_mode);
}

}  // namespace ckernel
