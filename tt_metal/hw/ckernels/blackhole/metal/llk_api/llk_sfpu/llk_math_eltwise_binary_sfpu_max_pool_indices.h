// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_max_pool_indices.h"

namespace ckernel {

template <bool APPROXIMATE, ckernel::DataLayout layout = ckernel::DataLayout::TILE>
inline void llk_math_eltwise_binary_sfpu_max_pool_with_indices_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::max_pool_with_indices, APPROXIMATE>(
        sfpu::init_max_pool_with_indices<APPROXIMATE, layout>);
}

template <
    bool APPROXIMATE,
    bool is_fp32_dest_acc_en,
    int num_rows,
    int ITERATIONS = 8,
    ckernel::DataLayout layout = ckernel::DataLayout::TILE>
inline void llk_math_eltwise_binary_sfpu_max_pool_with_indices(
    uint dst_index, uint32_t idx_index, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_max_pool_with_indices<APPROXIMATE, is_fp32_dest_acc_en, num_rows, ITERATIONS, layout>,
        dst_index,
        idx_index,
        vector_mode);
}

}  // namespace ckernel
