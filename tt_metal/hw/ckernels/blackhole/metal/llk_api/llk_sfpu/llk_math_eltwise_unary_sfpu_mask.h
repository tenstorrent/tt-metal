// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_mask.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mask_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::mask, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mask_posinf(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_mask_posinf<APPROXIMATE>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mask(
    uint dst_index, DataFormat data_format, int vector_mode = (int)VectorMode::RC) {
    if (data_format == DataFormat::Float16_b || data_format == DataFormat::Float16) {
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_mask<APPROXIMATE>, dst_index, vector_mode);
    } else if (data_format == DataFormat::Int32) {
        _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_int_mask<APPROXIMATE>, dst_index, vector_mode);
    }
}

}  // namespace ckernel
