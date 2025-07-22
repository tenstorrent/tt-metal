// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_quant.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_quant_int32_init(const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::quant_int32, APPROXIMATE>(
        sfpu::quant_init<APPROXIMATE>,
        zero_point);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_quant_int32(uint dst_index0, uint dst_index1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_quant_int32<APPROXIMATE>, dst_index0, dst_index1, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_requant_int32_init(const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::requant_int32, APPROXIMATE>(
        sfpu::quant_init<APPROXIMATE>,
        zero_point);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_requant_int32(uint dst_index0, uint dst_index1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_requant_int32<APPROXIMATE>, dst_index0, dst_index1, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_dequant_int32_init(const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::dequant_int32, APPROXIMATE>(
        sfpu::quant_init<APPROXIMATE>,
        zero_point);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_dequant_int32(uint dst_index0, uint dst_index1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_dequant_int32<APPROXIMATE>, dst_index0, dst_index1, vector_mode);
}

}  // namespace ckernel
