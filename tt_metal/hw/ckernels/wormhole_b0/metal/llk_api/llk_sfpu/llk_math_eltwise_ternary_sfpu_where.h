// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE, DataFormat data_format>
inline void llk_math_eltwise_ternary_sfpu_where(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::_calculate_where_<APPROX_MODE, data_format, 8>,
        dst_index0,
        dst_index1,
        dst_index2,
        odst,
        vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_ternary_sfpu_where_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();
    ckernel::sfpu::_init_where_<APPROX_MODE>();
}
}  // namespace ckernel
