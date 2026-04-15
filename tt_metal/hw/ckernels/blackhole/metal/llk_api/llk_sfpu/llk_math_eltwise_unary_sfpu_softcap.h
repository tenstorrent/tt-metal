// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_softcap.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softcap_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softcap, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softcap(
    uint dst_index, std::uint32_t param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        [param0](const int iterations) {
            ckernel::sfpu::_calculate_softcap_<APPROXIMATE, ITERATIONS>(iterations, param0);
        },
        dst_index,
        vector_mode);
}

}  // namespace ckernel
