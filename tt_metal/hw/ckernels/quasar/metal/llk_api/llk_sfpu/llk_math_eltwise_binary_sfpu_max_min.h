// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "ckernel_sfpu_binary_max_min.h"

namespace ckernel {

// Shared init for the float/MX max and min ops; the underlying sfpu init is
// format- and op-agnostic, so a single entry point covers both.
inline void llk_math_eltwise_binary_sfpu_binary_max_min_init() {
    _llk_math_eltwise_sfpu_init_();
    sfpu::_init_binary_max_min_();
}

// Shared init for the int32 max and min ops.
inline void llk_math_eltwise_binary_sfpu_binary_max_min_int32_init() {
    _llk_math_eltwise_sfpu_init_();
    sfpu::_init_binary_max_min_();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_max(
    std::uint32_t dst_index0, std::uint32_t dst_index1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_max_min<DataFormat::Float32, true>, dst_index0, dst_index1, odst, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_max_int32(
    std::uint32_t dst_index0, std::uint32_t dst_index1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_max_min<DataFormat::Int32, true>, dst_index0, dst_index1, odst, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_min(
    std::uint32_t dst_index0, std::uint32_t dst_index1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_max_min<DataFormat::Float32, false>, dst_index0, dst_index1, odst, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_min_int32(
    std::uint32_t dst_index0, std::uint32_t dst_index1, std::uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_max_min<DataFormat::Int32, false>, dst_index0, dst_index1, odst, vector_mode);
}

}  // namespace ckernel
