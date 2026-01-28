// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_remainder_int32.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_remainder_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::remainder_int32, APPROXIMATE>(sfpu::remainder_int32_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_remainder_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_remainder_int32<APPROXIMATE, 8>, dst_index0, dst_index1, odst, vector_mode);
}

}  // namespace ckernel
