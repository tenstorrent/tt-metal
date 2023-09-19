/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_trigonometry.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
//tan
ALWI void tan_tile(uint32_t idst) {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_tan_op<true, SyncHalf>(idst)));
}

ALWI void tan_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::tan, true>() ));}
} // namespace ckernel
