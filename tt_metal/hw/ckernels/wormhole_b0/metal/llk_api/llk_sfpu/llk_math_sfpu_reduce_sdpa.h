// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_reduce_custom.h"

namespace ckernel {

inline void llk_math_sfpu_reduce_max_sdpa_init() {
    ckernel::sfpu::_init_reduce_max_col_subblock_4x2_<DataFormat::Float16_b>();
}

inline void llk_math_sfpu_reduce_max_sdpa(
    uint32_t dst_index, uint32_t block_height, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::
            _calculate_reduce_max_col_subblock_4x2_<PoolType::MAX, ReduceDim::REDUCE_COL, DataFormat::Float16_b>,
        dst_index,
        vector_mode,
        block_height);
}

inline void llk_math_sfpu_reduce_max_col_epilogue() { ckernel::sfpu::_reduce_max_col_subblock_4x2_epilogue_(); }

inline void llk_math_sfpu_reduce_max_load_initial_values() {
    ckernel::sfpu::sfpu_reduce_max_col_subblock_4x2_load_initial_values();
}

inline void llk_math_sfpu_reduce_max_prologue() { ckernel::sfpu::_reduce_max_col_subblock_4x2_prologue_(); }

}  // namespace ckernel
