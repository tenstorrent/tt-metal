// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_reduce.h"
#include "ckernel_instr_params.h"

namespace ckernel::sfpu {

template <PoolType pool_type, ReduceDim reduce_dim, DataFormat format>
inline void calculate_reduce(uint32_t ct_dim, uint32_t rt_dim) {
    _calculate_reduce_<pool_type, reduce_dim, format>(ct_dim, rt_dim);
}

template <PoolType pool_type, DataFormat format>
inline void init_reduce() {
    _init_reduce_<pool_type, format>();
}

}  // namespace ckernel::sfpu
