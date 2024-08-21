// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_typecast.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, uint32_t IN_DTYPE, uint32_t OUT_DTYPE>
inline void llk_math_eltwise_unary_sfpu_typecast(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float16_b && OUT_DTYPE == (uint32_t)DataFormat::UInt16) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_uint16<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt16 && OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_uint16_to_fp16b<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Int32 && OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_int32_to_fp16b<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float16_b && OUT_DTYPE == (uint32_t)DataFormat::Int32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_int32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float16_b && OUT_DTYPE == (uint32_t)DataFormat::Float32) {
        // no SFPU kernel needed, handled by packer
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float32 && OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp32_to_fp16b<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float32 && OUT_DTYPE == (uint32_t)DataFormat::UInt16) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_uint16<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt16 && OUT_DTYPE == (uint32_t)DataFormat::Float32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_uint16_to_fp32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float32 && OUT_DTYPE == (uint32_t)DataFormat::Int32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_int32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Int32 && OUT_DTYPE == (uint32_t)DataFormat::Float32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_int32_to_fp32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp8_b && OUT_DTYPE == (uint32_t)DataFormat::UInt16) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_uint16<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt16 && OUT_DTYPE == (uint32_t)DataFormat::Bfp8_b) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_uint16_to_fp16b<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp8_b && OUT_DTYPE == (uint32_t)DataFormat::Int32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_int32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Int32 && OUT_DTYPE == (uint32_t)DataFormat::Bfp8_b) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_int32_to_fp16b<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float16_b && OUT_DTYPE == (uint32_t)DataFormat::UInt32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_uint32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt32 && OUT_DTYPE == (uint32_t)DataFormat::Float16_b) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_uint32_to_fp16b<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Float32 && OUT_DTYPE == (uint32_t)DataFormat::UInt32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_uint32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt32 && OUT_DTYPE == (uint32_t)DataFormat::Float32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_uint32_to_fp32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::Bfp8_b && OUT_DTYPE == (uint32_t)DataFormat::UInt32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_fp16b_to_uint32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt32 && OUT_DTYPE == (uint32_t)DataFormat::Bfp8_b) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_uint32_to_fp16b<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
    else if constexpr (IN_DTYPE == (uint32_t)DataFormat::UInt16 && OUT_DTYPE == (uint32_t)DataFormat::UInt32) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_typecast_uint16_to_uint32<APPROXIMATE,8>,
            dst_index,
            vector_mode);
    }
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_typecast_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

}
