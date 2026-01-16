// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"

namespace ckernel {

template <bool APPROXIMATE, DataFormat data_format>
inline void llk_math_eltwise_ternary_sfpu_where(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_where_<APPROXIMATE, data_format, 8>,
        dst_index0,
        dst_index1,
        dst_index2,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_where_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();
    ckernel::sfpu::_init_where_<APPROXIMATE>();
}
}  // namespace ckernel
