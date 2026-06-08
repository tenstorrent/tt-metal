// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "ckernel_sfpu_rsqrt.h"

namespace ckernel {

template <[[maybe_unused]] bool APPROXIMATE, [[maybe_unused]] bool legacy_compat = false>
inline void llk_math_eltwise_unary_sfpu_rsqrt_init() {
    static_assert(legacy_compat == false, "Non-default legacy_compat (true) not supported in Quasar rsqrt");
    llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt>();
}

template <
    [[maybe_unused]] bool APPROXIMATE,
    [[maybe_unused]] bool is_fp32_dest_acc_en,
    [[maybe_unused]] bool FAST_APPROX = false,
    [[maybe_unused]] bool legacy_compat = false,
    int ITERATIONS = SFPU_ITERATIONS>
inline void llk_math_eltwise_unary_sfpu_rsqrt(uint dst_index) {
    static_assert(FAST_APPROX == false, "Non-default FAST_APPROX (true) not supported in Quasar rsqrt");
    static_assert(legacy_compat == false, "Non-default legacy_compat (true) not supported in Quasar rsqrt");
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_rsqrt, dst_index, ITERATIONS);
}

}  // namespace ckernel
