/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_gelu.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

ALWI void gelu_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_gelu_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void gelu_tile(uint32_t idst, bool fast_and_approx=true) {
    if (fast_and_approx) {
        MATH(( llk_math_eltwise_unary_sfpu_gelu<true, SyncHalf>(idst) ));
    } else {
        MATH(( llk_math_eltwise_unary_sfpu_gelu<false, SyncHalf>(idst) ));
    }
}


} // namespace ckernel
