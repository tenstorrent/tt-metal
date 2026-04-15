// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
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
    uint dst_index, uint32_t inv_cap_param, uint32_t cap_param, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_softcap<APPROXIMATE, ITERATIONS>,
        dst_index,
        vector_mode,
        inv_cap_param,
        cap_param);
}

}  // namespace ckernel
