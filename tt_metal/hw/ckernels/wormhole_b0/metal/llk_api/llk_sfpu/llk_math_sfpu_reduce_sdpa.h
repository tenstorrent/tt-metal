// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sigmoid.h"

namespace ckernel {

inline void llk_math_sfpu_reduce_max_sdpa_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reduce, false>(sfpu::_init_reduce_sdpa_<DataFormat::Float16_b>);
}

inline void llk_math_sfpu_reduce_max_sdpa(uint dst_index, int vector_mode = (int)VectorMode::RC_custom) {
    // _llk_math_eltwise_unary_sfpu_params_<false>(
    //     ckernel::sfpu::_calculate_reduce_sdpa_<PoolType::MAX, REDUCE_COL, DataFormat::Float16_b>,
    //     dst_index,
    //     vector_mode,
    //     1  /* block_height  (for plain testing 1 -> for sdpa implementation 4) */);

    ckernel::sfpu::_calculate_reduce_sdpa_<PoolType::MAX, REDUCE_COL, DataFormat::Float16_b>(1);
}

}  // namespace ckernel
