// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_reduce.h"
#include "ckernel_instr_params.h"

namespace ckernel::sfpu {

template <PoolType pool_type, ReduceDim reduce_dim, DataFormat format, uint32_t ct_dim, uint32_t rt_dim>
inline void calculate_reduce() {
    _calculate_reduce_<pool_type, reduce_dim, format>(ct_dim, rt_dim);
}

template <DataFormat format>
inline void init_reduce() {
    // Use the specified format for initialization
    _init_reduce_<format>();
}

}  // namespace ckernel::sfpu
