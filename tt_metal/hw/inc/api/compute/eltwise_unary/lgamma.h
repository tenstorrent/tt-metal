#pragma once
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_lgamma.h"
#endif
namespace ckernel {
ALWI void lgamma_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_lgamma<APPROX, DST_ACCUM_MODE>(idst))); }
ALWI void lgamma_tile_init() { MATH((llk_math_eltwise_unary_sfpu_lgamma_init<APPROX, DST_ACCUM_MODE>())); }
}  // namespace ckernel
