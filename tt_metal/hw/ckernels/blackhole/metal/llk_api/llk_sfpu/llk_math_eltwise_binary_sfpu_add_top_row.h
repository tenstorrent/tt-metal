// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_add_top_row.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_add_top_row_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::add_top_row, APPROXIMATE>(
        sfpu::init_add_top_row);
}

template <bool APPROXIMATE, DataFormat format>
inline void llk_math_eltwise_binary_sfpu_add_top_row(
    uint dst_index_in_0, uint dst_index_in_1, uint dst_index_out) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_add_top_row<format>,
        dst_index_in_0,
        dst_index_in_1,
        dst_index_out,
        (int)VectorMode::RC_custom);
}

}  // namespace ckernel

