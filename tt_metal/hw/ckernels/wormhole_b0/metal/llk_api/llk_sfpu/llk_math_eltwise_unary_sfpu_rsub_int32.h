// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_rsub_int32.h"

namespace ckernel {

inline void llk_math_eltwise_unary_sfpu_rsub_int32_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rsub_int32(uint dst_index, uint scalar, int vector_mode = (int)VectorMode::RC) {
    SFPU_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsub_scalar_int32,
        (APPROXIMATE, ITERATIONS),
        dst_index,
        vector_mode,
        scalar);
}

}  // namespace ckernel
