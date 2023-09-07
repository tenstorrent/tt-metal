#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logical_not_noti.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
//logical_not
ALWI void logical_not_unary_tile(uint32_t idst) {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_logical_not_unary_op<true, SyncHalf>(idst)));
}

ALWI void logical_not_unary_tile_init() {
    MATH((ckernel::sfpu::llk_math_eltwise_unary_sfpu_logical_not_unary_init<true>() ));
}
} // namespace ckernel
