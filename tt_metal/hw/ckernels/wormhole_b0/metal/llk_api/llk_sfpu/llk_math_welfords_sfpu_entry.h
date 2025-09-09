// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "llk_math_welfords_sfpu_entry.h"
#include "llk_math_welfords_sfpu_params.h"

namespace ckernel {

inline void llk_math_welfords_sfpu_init() { _llk_math_welfords_sfpu_init_(); }

template <
    uint32_t input_dst_index,
    uint32_t mean_dst_index,
    uint32_t m2_dst_index,
    bool reformat_dst_to_col_on_end,
    bool convert_M2_to_var_on_end>
inline void llk_math_welfords_sfpu(
    uint32_t current_row, uint32_t final_row, uint32_t num_skip_rows, uint32_t reciprocal_lut_ptr) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_welfords_llk_entry_,
        input_dst_index,
        mean_dst_index,
        m2_dst_index,
        current_row,
        final_row,
        num_skip_rows,
        reciprocal_lut_ptr,
        reformat_dst_to_col_on_end,
        convert_M2_to_var_on_end);
}

}  // namespace ckernel
