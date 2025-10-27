// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_reduce.h"

namespace ckernel {

template <bool APPROXIMATE, DataFormat format>
inline void llk_math_eltwise_unary_sfpu_reduce_sum_avg_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reduce_sum_avg, APPROXIMATE>(ckernel::sfpu::init_reduce<format>);
}

template <bool APPROXIMATE, PoolType pool_type, ReduceDim reduce_dim, DataFormat format>
inline void llk_math_eltwise_unary_sfpu_reduce_sum_avg(uint dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_reduce<pool_type, reduce_dim, format>,
        dst_index,
        vector_mode);
}

}  // namespace ckernel
