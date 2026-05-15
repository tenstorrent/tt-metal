// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "ckernel_sfpu_tanh.h"

namespace ckernel {

template <[[maybe_unused]] bool APPROXIMATE, [[maybe_unused]] bool EN_32BIT_DEST>
inline void llk_math_eltwise_unary_sfpu_tanh_init() {
    static_assert(EN_32BIT_DEST == false, "Non-default EN_32BIT_DEST not supported in quasar tanh");
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh>();
}

template <bool APPROXIMATE, [[maybe_unused]] bool EN_32BIT_DEST, int ITERATIONS = SFPU_ITERATIONS>
inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_index) {
    static_assert(EN_32BIT_DEST == false, "Non-default EN_32BIT_DEST not supported in quasar tanh");
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_tanh<APPROXIMATE>, dst_index, ITERATIONS);
}

}  // namespace ckernel
