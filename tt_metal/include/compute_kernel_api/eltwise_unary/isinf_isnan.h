/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_isinf_isnan.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
//isinf
ALWI void isinf_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_isinf<APPROX, SyncHalf>(idst)));
}

ALWI void isinf_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_isinf_init<APPROX>() ));
}

//isposinf
ALWI void isposinf_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_isposinf<APPROX, SyncHalf>(idst) ));
}

ALWI void isposinf_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_isposinf_init<APPROX>() ));
}

//isneginf
ALWI void isneginf_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_isneginf<APPROX, SyncHalf>(idst) ));
}

ALWI void isneginf_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_isneginf_init<APPROX>() ));
}

//isnan
ALWI void isnan_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_isnan<APPROX, SyncHalf>(idst) ));
}

ALWI void isnan_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_isnan_init<APPROX>() ));
}

//isfinite
ALWI void isfinite_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_isfinite<APPROX, SyncHalf>(idst) ));
}

ALWI void isfinite_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_isfinite_init<APPROX>() ));
}
} // namespace ckernel
