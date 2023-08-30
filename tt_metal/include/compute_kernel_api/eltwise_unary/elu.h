#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_elu.h"  // tt_metal/src/ckernels/grayskull/common/inc/ckernel_sfpu_erf_erfc.h
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

//elu : y = relu(x) + slope*(exp(x) - 1)*(x <= 0 );
ALWI void elu_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_elu<true, SyncHalf>(idst,param0) ));
}

ALWI void elu_tile_init() {
  MATH(( llk_math_eltwise_unary_sfpu_elu_init<true>() ));
}

} // namespace ckernel
