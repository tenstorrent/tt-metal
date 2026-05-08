// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_nextafter.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_nextafter_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>();
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT = DataFormat::Float32, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_nextafter(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_sfpu_nextafter<APPROXIMATE, ITERATIONS, DATA_FORMAT>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel
