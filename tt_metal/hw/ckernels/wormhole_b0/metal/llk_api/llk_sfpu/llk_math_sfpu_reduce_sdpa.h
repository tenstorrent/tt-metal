// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sigmoid.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_sfpu_reduce_max_sdpa_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reduce, APPROXIMATE>(sfpu::_init_reduce_sdpa_<DataFormat::Float16_b>);
}

template <bool APPROXIMATE>
inline void llk_math_sfpu_reduce_max_sdpa(uint dst_index, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_reduce_sdpa_<PoolType::MAX, REDUCE_COL, DataFormat::Float16_b>,
        dst_index,
        vector_mode,
        4);
}

}  // namespace ckernel
