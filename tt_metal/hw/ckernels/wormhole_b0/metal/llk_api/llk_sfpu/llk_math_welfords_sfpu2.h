// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "llk_math_welfords_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"

namespace ckernel {

inline void llk_math_welfords_sfpu_init() {
    _llk_math_welfords_sfpu_init_();
}

inline void llk_math_welfords_sfpu(
    uint32_t dst_index0,
    uint32_t dst_index1,
    uint32_t dst_index2,
    uint32_t current_sample,
    uint32_t final_sample,
    uint32_t reformat_dst = 1,
    uint32_t skip_n_samples = 0) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_welfords_llk_entry_,
        dst_index0,
        dst_index1,
        dst_index2,
        current_sample,
        final_sample,
        reformat_dst,
        skip_n_samples);
}

}  // namespace ckernel
