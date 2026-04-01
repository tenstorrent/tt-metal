// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_atan2.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_binary_sfpu_atan2_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(
        sfpu::calculate_sfpu_atan2_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void llk_math_eltwise_binary_sfpu_atan2(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_sfpu_atan2<APPROXIMATE, ITERATIONS, is_fp32_dest_acc_en>, dst_index0, dst_index1, odst, vector_mode);
}

}  // namespace ckernel
