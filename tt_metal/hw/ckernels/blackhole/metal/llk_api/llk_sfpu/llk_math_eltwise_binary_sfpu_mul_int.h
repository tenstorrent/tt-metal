// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_mul_int32.h"
#include "llk_defs.h"

namespace ckernel {

template <bool APPROXIMATE, DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_mul_int_init() {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for mul_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::UInt16) {
        llk_math_eltwise_binary_sfpu_init<SfpuType::mul_uint16, APPROXIMATE>(sfpu::_init_mul_int_<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise)>);
    } else {
        llk_math_eltwise_binary_sfpu_init<SfpuType::mul_int32, APPROXIMATE>(sfpu::mul_int32_init<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise)>);
    }
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_mul_int(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for mul_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::UInt16) {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            sfpu::_mul_int_<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise), ITERATIONS>, dst_index0, dst_index1, odst, vector_mode);
    } else {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            sfpu::mul_int32<(APPROXIMATE ? ApproximationMode::Fast : ApproximationMode::Precise)>, dst_index0, dst_index1, odst, vector_mode);
    }
}

}  // namespace ckernel