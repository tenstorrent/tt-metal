#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_isinf_isnan.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
//isinf
ALWI void isinf_tile(uint32_t idst) {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_isinf<true, SyncHalf>(idst)));
}

ALWI void isinf_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::isinf, true>() ));
}

//isposinf
ALWI void isposinf_tile(uint32_t idst) {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_isposinf<true, SyncHalf>(idst) ));
}

ALWI void isposinf_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::isposinf, true>() ));
}

//isneginf
ALWI void isneginf_tile(uint32_t idst) {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_isneginf<true, SyncHalf>(idst) ));
}

ALWI void isneginf_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::isneginf, true>() ));
}

//isnan
ALWI void isnan_tile(uint32_t idst) {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_isnan<true, SyncHalf>(idst) ));
}

ALWI void isnan_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::isnan, true>() ));
}
} // namespace ckernel
