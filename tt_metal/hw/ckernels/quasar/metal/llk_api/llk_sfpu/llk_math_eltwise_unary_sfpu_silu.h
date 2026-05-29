// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "ckernel_sfpu_silu.h"

namespace ckernel {

template <[[maybe_unused]] bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_silu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::silu>();
}

template <
    [[maybe_unused]] bool APPROXIMATE,
    [[maybe_unused]] bool is_fp32_dest_acc_en,
    int ITERATIONS = SFPU_ITERATIONS>
inline void llk_math_eltwise_unary_sfpu_silu(uint dst_index) {
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_silu, dst_index, ITERATIONS);
}

}  // namespace ckernel
