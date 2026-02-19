// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary_fmod.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_fmod_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::fmod_int32, APPROXIMATE>(sfpu::fmod_int32_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_fmod_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_fmod_int32<APPROXIMATE, 8>, dst_index0, dst_index1, odst, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_fmod_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::fmod_binary_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binary_fmod(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_sfpu_binary_fmod<APPROXIMATE, 8, is_fp32_dest_acc_en>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel
