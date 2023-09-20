/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_i0.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
//i0
ALWI void i0_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_i0_op<APPROX, SyncHalf>(idst)));
}

ALWI void i0_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_i0_init<APPROX>() ));
}
} // namespace ckernel
