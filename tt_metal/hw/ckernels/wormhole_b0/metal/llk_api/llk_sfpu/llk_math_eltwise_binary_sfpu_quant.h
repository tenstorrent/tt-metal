// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
// #include "ckernel_sfpu_.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_quant_int32(
    uint dst_index_a, uint dst_index_b, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_binary_sfpu<SfpuType::quant_int32, APPROXIMATE>(dst_index_a, dst_index_b, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_quant_int32_init(const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::quant_int32, APPROXIMATE>(zero_point);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_requant_int32(
    uint dst_index_a, uint dst_index_b, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_binary_sfpu<SfpuType::requant_int32, APPROXIMATE>(dst_index_a, dst_index_b, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_requant_int32_init(const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::requant_int32, APPROXIMATE>(zero_point);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_dequant_int32(
    uint dst_index_a, uint dst_index_b, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_binary_sfpu<SfpuType::dequant_int32, APPROXIMATE>(dst_index_a, dst_index_b, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_dequant_int32_init(const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::dequant_int32, APPROXIMATE>(zero_point);
}


}  // namespace ckernel
