// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_max_pool_with_indices_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::max_pool_with_indices, APPROXIMATE>(sfpu::_init_max_pool_with_indices_);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, int num_rows>
inline void llk_math_eltwise_binary_sfpu_max_pool_with_indices(
    uint dst_index, uint32_t idx_index, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_calculate_max_pool_with_indices_<APPROXIMATE, is_fp32_dest_acc_en, num_rows, 8>,
        dst_index,
        idx_index,
        0, /* unused */
        vector_mode);
}

}  // namespace ckernel
