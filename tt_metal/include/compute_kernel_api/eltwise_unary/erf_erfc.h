/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_erf_erfc.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/************** ERF *****************/

ALWI void erf_tile_init() { MATH((llk_math_eltwise_unary_sfpu_erf_init<APPROX>())); }

ALWI void erf_tile(uint32_t idst, bool fast_and_approx = true) {
    if (fast_and_approx) {
        MATH(( llk_math_eltwise_unary_sfpu_erf<true, SyncHalf>(idst) ));
    } else {
        MATH(( llk_math_eltwise_unary_sfpu_erf<false, SyncHalf>(idst) ));
    }
}

/************** ERFC *****************/

ALWI void erfc_tile_init() { MATH((llk_math_eltwise_unary_sfpu_erfc_init<APPROX>())); }

ALWI void erfc_tile(uint32_t idst, bool fast_and_approx = true) {
    if (fast_and_approx) {
        MATH(( llk_math_eltwise_unary_sfpu_erfc<true, SyncHalf>(idst) ));
    } else {
        MATH(( llk_math_eltwise_unary_sfpu_erfc<false, SyncHalf>(idst) ));
    }
}

}  // namespace ckernel
