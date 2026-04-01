// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_atan2.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise atan2 operation y = atan2(x0, x1) with x0 as the first operand and x1 as the second operand.
 * Output overwrites the destination tile at `odst` in DST.
 */
// clang-format on
ALWI void atan2_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_atan2<APPROX, DST_ACCUM_MODE, 8>(idst0, idst1, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void atan2_binary_tile_init() { MATH((llk_math_eltwise_binary_sfpu_atan2_init<APPROX, DST_ACCUM_MODE>())); }

}  // namespace ckernel
