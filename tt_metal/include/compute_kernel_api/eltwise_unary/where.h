// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu_where.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// PLEASE SOMEONE CHANGE BELOW LINES
// clang-format off
/**
 */
// clang-format on
ALWI void where_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2) {
    MATH((llk_math_eltwise_ternary_sfpu_where<APPROX>(idst0, idst1, idst2)));
}

// clang-format off
/**
*/
// clang-format on
ALWI void where_fp32_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2) {
    MATH((llk_math_eltwise_ternary_sfpu_where_fp32<APPROX>(idst0, idst1, idst2)));
}

// clang-format off
/**
*/
// clang-format on
ALWI void where_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2) {
    MATH((llk_math_eltwise_ternary_sfpu_where_int32<APPROX>(idst0, idst1, idst2)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void where_tile_init() { MATH((llk_math_eltwise_ternary_sfpu_where_init<APPROX>())); }

}  // namespace ckernel
