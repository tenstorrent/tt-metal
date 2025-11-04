// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sigmoid.h"

namespace ckernel {

inline void llk_math_sfpu_reduce_max_sdpa_init(uint32_t num_cols) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reduce, false>(
        sfpu::_init_reduce_sdpa_<DataFormat::Float16_b>, num_cols);
}

inline void llk_math_sfpu_reduce_max_sdpa(
    uint32_t dst_index, uint32_t block_height, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::_calculate_reduce_sdpa_<PoolType::MAX, REDUCE_COL, DataFormat::Float16_b>,
        dst_index,
        vector_mode,
        block_height);
}

}  // namespace ckernel
