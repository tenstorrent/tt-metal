/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_trigonometry.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

//sine
ALWI void sin_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_sine_init<APPROX>()));
}

ALWI void sin_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sine_op<APPROX, SyncHalf>(idst)));
}

//cosine
ALWI void cos_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_cosine_init<APPROX>()));
}

ALWI void cos_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_cosine_op<APPROX, SyncHalf>(idst)));
}

//tan
ALWI void tan_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_tan_init<APPROX>()));
}

ALWI void tan_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_tan_op<APPROX, SyncHalf>(idst)));
}
} // namespace ckernel
