/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_i0.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
//logical_not
ALWI void i0_tile(uint32_t idst) {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_i0_op<true, SyncHalf>(idst)));
}

ALWI void i0_tile_init() {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_i0_init<true>() ));
}
} // namespace ckernel
