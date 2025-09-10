// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_max_pool_indices.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_max_pool_with_indices_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::max_pool_with_indices, APPROXIMATE>(
        sfpu::init_max_pool_with_indices<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, int num_rows>
inline void llk_math_eltwise_binary_sfpu_max_pool_with_indices(
    uint dst_index, uint32_t idx_index, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_max_pool_with_indices<APPROXIMATE, is_fp32_dest_acc_en, num_rows>,
        dst_index,
        idx_index,
        vector_mode);
}

}  // namespace ckernel
