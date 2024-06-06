// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_typecast.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, uint32_t OUT_DTYPE>
inline void llk_math_eltwise_unary_sfpu_typecast(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    if constexpr (OUT_DTYPE == (uint32_t)DataFormat::UInt16) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_uint16<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (OUT_DTYPE == (uint32_t)DataFormat::UInt32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_uint32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_uint16_to_fp16b<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_typecast_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

}
