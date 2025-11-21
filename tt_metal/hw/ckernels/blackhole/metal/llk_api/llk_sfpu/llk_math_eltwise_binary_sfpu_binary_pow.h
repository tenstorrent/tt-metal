// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary_pow.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_pow_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(ckernel::sfpu::sfpu_binary_pow_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binary_pow(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary_pow<APPROXIMATE, 8, is_fp32_dest_acc_en>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel
