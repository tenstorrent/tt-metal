// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "llk_math_welfords_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"

namespace ckernel {

inline void llk_math_welfords_sfpu_init() { _llk_math_welfords_sfpu_init_(); }

inline void
llk_math_welfords_sfpu<uint32_t input_dst_index, uint32_t mean_dst_index, uint32_t m2_dst_index, uint32_t reformat_dst>(
    uint32_t current_row, uint32_t final_row, uint32_t num_skip_rows) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_welfords_llk_entry_,
        input_dst_index,
        mean_dst_index,
        m2_dst_index,
        current_row,
        final_row,
        reformat_dst,
        num_skip_rows);
}

}  // namespace ckernel
