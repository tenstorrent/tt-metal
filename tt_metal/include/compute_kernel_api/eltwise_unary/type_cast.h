// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_typecast.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

ALWI void to_uint16_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_to_uint16_init<APPROX>() ));
}

ALWI void to_uint16_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_to_uint16<APPROX, SyncHalf>(idst) ));
}

ALWI void to_uint32_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_to_uint32_init<APPROX>() ));
}

ALWI void to_uint32_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_to_uint32<APPROX, SyncHalf>(idst) ));
}
