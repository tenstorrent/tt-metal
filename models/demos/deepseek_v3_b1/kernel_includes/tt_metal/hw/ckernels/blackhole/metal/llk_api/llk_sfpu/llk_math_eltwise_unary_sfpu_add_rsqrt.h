// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_add_rsqrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// add_rsqrt is a deepseek-vendored op that is not part of the canonical SFPU
// tree, so it keeps its own thin wrapper rather than living in api/compute. The
// body uses the shared unary SFPU macros to match the post-refactor convention.

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_add_rsqrt_init() {
    SFPU_INIT_CB(rsqrt, sfpu::init_add_rsqrt, (APPROXIMATE));
}

template <bool APPROXIMATE, bool fp32_dest_acc_en, bool FAST_APPROX, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_add_rsqrt(
    uint dst_index, uint32_t param0, VectorMode vector_mode = VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_add_rsqrt,
        (APPROXIMATE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX),
        dst_index,
        vector_mode,
        param0);
}

}  // namespace ckernel
