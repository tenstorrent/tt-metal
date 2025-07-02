// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_sub_int_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <
    bool APPROXIMATE,
    int ITERATIONS = 8,
    InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_sub_int(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_sub_int_<APPROXIMATE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>,
        dst_index0,
        dst_index1,
        vector_mode);
}

}  // namespace ckernel
