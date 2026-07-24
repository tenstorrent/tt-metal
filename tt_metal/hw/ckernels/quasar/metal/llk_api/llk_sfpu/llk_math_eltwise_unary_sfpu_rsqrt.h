// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_rsqrt.h"

namespace ckernel {

template <bool APPROXIMATE, [[maybe_unused]] bool legacy_compat = false>
inline void llk_math_eltwise_unary_sfpu_rsqrt_init() {
    static_assert(legacy_compat == false, "Non-default legacy_compat (true) not supported in Quasar rsqrt");
    // Run the global SFPU config init followed by rsqrt_init, which programs the full-precision
    // SQRT_23-bits constants; the bare init<> form alone would skip that op setup.
    llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt>(sfpu::rsqrt_init<APPROXIMATE, legacy_compat>);
}

template <
    bool APPROXIMATE,
    bool is_fp32_dest_acc_en,
    [[maybe_unused]] bool FAST_APPROX = false,
    [[maybe_unused]] bool legacy_compat = false,
    int ITERATIONS = SFPU_ITERATIONS>
inline void llk_math_eltwise_unary_sfpu_rsqrt(uint dst_index) {
    static_assert(FAST_APPROX == false, "Non-default FAST_APPROX (true) not supported in Quasar rsqrt");
    static_assert(legacy_compat == false, "Non-default legacy_compat (true) not supported in Quasar rsqrt");
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROXIMATE, ITERATIONS, is_fp32_dest_acc_en, FAST_APPROX, legacy_compat),
        dst_index,
        VectorMode::RC);
}

}  // namespace ckernel
