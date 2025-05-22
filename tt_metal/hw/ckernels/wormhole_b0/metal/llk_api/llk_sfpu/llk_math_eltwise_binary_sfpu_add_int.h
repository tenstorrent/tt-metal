// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_add_int_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE, bool SIGN_MAGNITUDE_FORMAT, InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32>
inline void llk_math_eltwise_binary_sfpu_add_int(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::_add_int_<APPROXIMATE, SIGN_MAGNITUDE_FORMAT, INSTRUCTION_MODE>,
        dst_index0,
        dst_index1,
        vector_mode);
}

}  // namespace ckernel
