// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_reduce.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE, PoolType pool_type, DataFormat format>
inline void llk_math_eltwise_unary_sfpu_reduce_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reduce, APPROX_MODE>(sfpu::init_reduce<pool_type, format>);
}

template <ckernel::ApproximationMode APPROX_MODE, PoolType pool_type, ReduceDim reduce_dim, DataFormat format>
inline void llk_math_eltwise_unary_sfpu_reduce(uint dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        sfpu::calculate_reduce<pool_type, reduce_dim, format>, dst_index, vector_mode);
}

}  // namespace ckernel
