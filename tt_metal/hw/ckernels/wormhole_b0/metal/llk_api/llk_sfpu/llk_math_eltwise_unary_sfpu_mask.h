// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_mask.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_mask_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::mask, APPROX_MODE>();
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_mask_posinf(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_mask_posinf<APPROX_MODE>, dst_index, vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_mask(
    uint dst_index, DataFormat data_format, int vector_mode = (int)VectorMode::RC) {
    if (data_format == DataFormat::Float16_b || data_format == DataFormat::Float16) {
        _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
            ckernel::sfpu::calculate_mask<APPROX_MODE>, dst_index, vector_mode);
    } else if (data_format == DataFormat::Int32) {
        _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
            ckernel::sfpu::calculate_int_mask<APPROX_MODE>, dst_index, vector_mode);
    }
}

}  // namespace ckernel
