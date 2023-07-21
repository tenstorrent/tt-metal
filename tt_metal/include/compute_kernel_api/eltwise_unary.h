#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu.h"
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
ALWI void gelu_tile(uint32_t idst,bool fast_and_approx=true) {
  if (fast_and_approx) {
    MATH(( llk_math_eltwise_unary_sfpu_gelu<true, SyncHalf>(idst) ));
  } else {
    MATH(( llk_math_eltwise_unary_sfpu_gelu<false, SyncHalf>(idst) ));
  }
}

ALWI void recip_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_reciprocal_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void recip_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_reciprocal<APPROX, SyncHalf>(idst) ));
}

ALWI void exp_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_exponential_init<APPROX>() ));
}

/**
 * Performs element-wise computation of exponential on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void exp_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_exponential<APPROX, SyncHalf>(idst) ));
}

ALWI void sqrt_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sqrt_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void sqrt_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sqrt<APPROX, SyncHalf>(idst) ));
}

ALWI void sigmoid_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid_init<APPROX>() )); // TODO(AP): move out init
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void sigmoid_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid<APPROX, SyncHalf>(idst) ));
}

ALWI void log_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_log_init<APPROX>() )); // TODO(AP): move out init
}

/**
 *  Please refer to documentation for log_tile.
 */
ALWI void log_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_log<APPROX, SyncHalf>(idst) ));
}

ALWI void log_with_base_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base_init<APPROX>()));  // TODO(AP): move out init
}

/**
 *  Please refer to documentation for log_with_base_tile.
 */
ALWI void log_with_base_tile(uint32_t idst,uint32_t base_scale) {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base<APPROX, SyncHalf>(idst, base_scale)));
}

ALWI void tanh_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_tanh_init<APPROX>() )); // TODO(AP): move out init
}

ALWI void sin_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sin_init<APPROX>() ));
}

ALWI void cos_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_cos_init<APPROX>() ));
}

/**
 *  Please refer to documentation for exp_tile.
 */
ALWI void tanh_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_tanh<APPROX, SyncHalf>(idst) ));
}

ALWI void sin_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sin<APPROX, SyncHalf>(idst) ));
}

ALWI void cos_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_cos<APPROX, SyncHalf>(idst) ));
}

} // namespace ckernel
