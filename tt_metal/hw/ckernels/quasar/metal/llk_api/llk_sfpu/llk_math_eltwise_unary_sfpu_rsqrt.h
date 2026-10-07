// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_rsqrt.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rsqrt_init() {
    // Run the global SFPU config init followed by rsqrt_init, which programs the full-precision
    // SQRT_23-bits constants; the bare init<> form alone would skip that op setup.
    llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt>(sfpu::rsqrt_init<APPROXIMATE>);
}

template <
    bool APPROXIMATE,
    bool is_fp32_dest_acc_en,
    bool FAST_APPROX /*maybe_unused*/ = false,
    int ITERATIONS = SFPU_ITERATIONS>
inline void llk_math_eltwise_unary_sfpu_rsqrt(uint dst_index) {
    static_assert(FAST_APPROX == false, "Non-default FAST_APPROX (true) not supported in Quasar rsqrt");
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROXIMATE, ITERATIONS, is_fp32_dest_acc_en, FAST_APPROX),
        dst_index,
        VectorMode::RC);
}

}  // namespace ckernel
