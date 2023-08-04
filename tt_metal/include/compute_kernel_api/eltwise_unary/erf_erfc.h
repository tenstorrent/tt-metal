#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_erf_erfc.h"  // tt_metal/src/ckernels/grayskull/common/inc/ckernel_sfpu_erf_erfc.h
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/************** ERF *****************/

ALWI void erf_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::erf, APPROX>())); }

ALWI void erf_tile(uint32_t idst, bool fast_and_approx = true) {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_erf<APPROX, SyncHalf>(idst, (uint32_t)fast_and_approx)));
}

/************** ERFC *****************/

ALWI void erfc_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::erfc, APPROX>())); }

ALWI void erfc_tile(uint32_t idst, bool fast_and_approx = true) {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_erfc<APPROX, SyncHalf>(idst, (uint32_t)fast_and_approx)));
}

}  // namespace ckernel
