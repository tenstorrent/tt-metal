// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "llk_math_welfords_sfpu.h"

namespace ckernel {

template <>
inline void llk_math_welfords_sfpu_init() {
    _llk_math_welfords_sfpu_init_();
}

inline void llk_math_welfords_sfpu(
    uint32_t dst_index0,
    uint32_t dst_index1,
    uint32_t dst_index2,
    uint32_t start_N,
    uint32_t end_N,
    uint32_t last_run) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_welfords_, dst_index0, dst_index1, dst_index2, start_N, end_N, last_run);
}

}  // namespace ckernel
