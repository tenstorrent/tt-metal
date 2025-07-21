// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_fmod.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_fmod_init(uint param0, uint param1) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::fmod, APPROXIMATE>(sfpu::init_fmod<APPROXIMATE>, param0, param1);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_fmod(
    uint dst_index, uint param0, uint param1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_fmod<APPROXIMATE>, dst_index, vector_mode, param0, param1);
}

}  // namespace ckernel
