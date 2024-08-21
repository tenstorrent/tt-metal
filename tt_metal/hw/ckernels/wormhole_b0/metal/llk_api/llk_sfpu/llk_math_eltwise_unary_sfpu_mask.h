// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_mask.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mask_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::mask, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mask(uint dst_index, DataFormat data_format, int vector_mode = (int)VectorMode::RC) {
    if (data_format == DataFormat::Float16_b || data_format == DataFormat::Float16) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_mask<APPROXIMATE>, dst_index, vector_mode);
    } else if (data_format == DataFormat::Int32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_int_mask<APPROXIMATE>, dst_index, vector_mode);
    }
}

}
