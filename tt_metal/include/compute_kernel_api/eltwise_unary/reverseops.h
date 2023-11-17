// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_reverseops.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

//rpow: implemented as a composite operator
//rpow(a,k) = k**(a)

//RDIV : rdiv(x,y) = y/x
//implemented as tied multiply operator

//RSUB : rsub(x,y) = y-x
ALWI void rsub_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_rsub<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void rsub_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_rsub_init<APPROX>() ));
}

} // namespace ckerne
