// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "chlkc_list.h"
#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_debug.h"
#include "hostdevcommon/kernel_structs.h"
#include "risc_attribs.h"

#define SYNC SyncHalf

#define ALWI inline __attribute__((always_inline))

#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "llk_math_matmul_api.h"
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_binary_api.h"
#include "llk_math_unary_sfpu_api.h"
#include "llk_math_reduce_api.h"
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#include "llk_io_pack.h"
#define PACK(x) x
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_A_api.h"
#include "llk_unpack_AB_api.h"
#include "llk_unpack_reduce_api.h"
#include "llk_unpack_tilize_api.h"
#include "llk_unpack_untilize_api.h"
#include "llk_io_unpack.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif


namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void rsqrt_tile_init(bool fast_and_approx=true) {
    if (fast_and_approx) {
        MATH(( llk_math_eltwise_unary_sfpu_rsqrt_init<true>() ));
    } else {
        MATH(( llk_math_eltwise_unary_sfpu_rsqrt_init<false>() ));
    }
}

/**
 * Performs element-wise computation of reciprocal sqrt on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | fast_and_approx | Computation to be done faster and approximate                              | bool     |                                                       | False    |
 */
ALWI void rsqrt_tile(uint32_t idst, bool fast_and_approx=true) {
  if (fast_and_approx) {
    MATH(( llk_math_eltwise_unary_sfpu_rsqrt<true, SyncHalf>(idst) ));
  } else {
    MATH(( llk_math_eltwise_unary_sfpu_rsqrt<false, SyncHalf>(idst) ));
  }
}


/**
 * Please refer to documentation for any_init.
 */
ALWI void sigmoid_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid_init<APPROX>() )); // TODO(AP): move out init
}

/**
 * Performs element-wise computation of sigmoid on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void sigmoid_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid<APPROX, SyncHalf>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void log_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_log_init<APPROX>() )); // TODO(AP): move out init
}

/**
 * Performs element-wise computation of logarithm on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void log_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_log<APPROX, SyncHalf>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void log_with_base_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base_init<APPROX>()));  // TODO(AP): move out init
}

/**
 * Performs element-wise computation of logarithm with a specified base on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | base_scale      | The log base                                                               | uint32_t |  Postive integers                                     | True     |
 */
ALWI void log_with_base_tile(uint32_t idst,uint32_t base_scale) {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base<APPROX, SyncHalf>(idst, base_scale)));
}

//TODO: Move to trigonometry.h
/**
 * Please refer to documentation for any_init.
 */
ALWI void tanh_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_tanh_init<APPROX>() )); // TODO(AP): move out init
}

//TODO: Move to trigonometry.h
/**
 * Performs element-wise computation of tanh on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void tanh_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_tanh<APPROX, SyncHalf>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void signbit_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_signbit_init<APPROX>() ));
}

/**
 * Sets the sign bit of each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to modify the sign bit of     | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void signbit_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_signbit<APPROX, SyncHalf>(idst) ));
}



/**
 * Performs element-wise computation of absolute value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void abs_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_abs<APPROX, SyncHalf>(idst) ));
}


/**
 * Please refer to documentation for any_init.
 */
ALWI void abs_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_abs_init<APPROX>() ));
}

/**
 * Will store in the output of the compute core the signum of the tile.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void sign_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_sign<APPROX, SyncHalf>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void sign_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sign_init<APPROX>() ));
}

/**
 * Performs element-wise computation of square value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void square_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_square<APPROX, SyncHalf>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void square_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_square_init<APPROX>() ));
}

//compare to zero operators




/**
 * Will store in the output of the compute core True if each element of a tile is less than zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */

ALWI void ltz_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_ltz<APPROX, SyncHalf>(idst) ));
}


/**
 * Please refer to documentation for any_init.
 */
ALWI void ltz_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_ltz_init<APPROX>() ));
}


/**
 * Will store in the output of the compute core True if each element of a equal to zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */

ALWI void eqz_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_eqz<APPROX,SyncHalf>(idst) ));
}


/**
 * Please refer to documentation for any_init.
 */
ALWI void eqz_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_eqz_init<APPROX>() ));
}

/**
 * Will store in the output of the compute core True if each element is less than or equal to zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */

ALWI void lez_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_lez<APPROX, SyncHalf>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void lez_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_lez_init<APPROX>() ));
}

/**
 * Will store in the output of the compute core True if each element is greater than zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */

ALWI void gtz_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_gtz<APPROX, SyncHalf>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void gtz_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_gtz_init<APPROX>() ));
}

/**
 * Will store in the output of the compute core True if each element is not equal to zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void nez_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_nez<APPROX, SyncHalf>(idst) ));
}


/**
 * Please refer to documentation for any_init.
 */
ALWI void nez_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_nez_init<APPROX>() ));
}

/**
 * Will store in the output of the compute core True if each element is greater than or equal to zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void gez_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_gez<APPROX, SyncHalf>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void gez_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_gez_init<APPROX>() ));
}



//POWER : y = x^(const param0)
/**
 * Performs element-wise computation of power operation (x ^(const param0)) value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value of the exponent in the power operation                           | uint32_t |                                                       | True     |
 */
ALWI void power_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_power<APPROX, SyncHalf>(idst,param0) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void power_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_power_init<APPROX>() ));
}

//MAX : y = max(idst0, idst1)
/**
 * Performs element-wise computation of max value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * TODO: fix idst1.
 * currently idst1 is not used and (idst0 + 1) is used for max.
 * because don't know how to use 2 dst register with sfpu.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void max_tile(uint32_t idst0, uint32_t idst1) {
    MATH(( llk_math_eltwise_unary_sfpu_max<APPROX, SyncHalf>(idst0) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void max_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_max_init<APPROX>() ));
}

//exp2 : y = 2 ^ x  ==> [y = exp(x * log(2))]
/**
 * Performs element-wise computation of 2^x value where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void exp2_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_exp2<true, SyncHalf>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void exp2_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_exp2_init<true>() ));
}

//heaviside : y = 0 if x < 0 , 1 if x > 0 , else value
/**
 * Performs element-wise computation of:  y = 0 if x < 0 , 1 if x > 0 , y= value  where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value the output is if the input is greater than 0                     | uint32_t |                                                       | True     |
 */
ALWI void heaviside_tile(uint32_t idst,uint32_t param0) {
    MATH(( llk_math_eltwise_unary_sfpu_heaviside<APPROX, SyncHalf>(idst,param0) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void heaviside_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_heaviside_init<APPROX>() ));
}

//expm1 : (exp(x) - 1)
/**
 * Performs element-wise computation of exp(x) - 1, v where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void expm1_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_expm1<true, SyncHalf>(idst) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void expm1_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_expm1_init<true>() ));
}

//TODO: move to trigonometry.h
/**
 * Performs element-wise computation of arcsine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void asin_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_asin<true, SyncHalf>(idst) ));
}

//TODO: move to trigonometry.h
/**
 * Please refer to documentation for any_init.
 */
ALWI void asin_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_asin_init<true>() ));
}

//TODO: move to trigonometry.h
/**
 * Performs element-wise computation of arctan on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void atan_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_atan<true, SyncHalf>(idst) ));
}

//TODO: move to trigonometry.h
/**
 * Please refer to documentation for any_init.
 */
ALWI void atan_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_atan_init<true>() ));
}

//TODO: move to trigonometry.h
/**
 * Performs element-wise computation of arccossine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void acos_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_acos<true, SyncHalf>(idst) ));
}

//TODO: move to trigonometry.h
/**
 * Please refer to documentation for any_init.
 */
ALWI void acos_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_acos_init<true>() ));
}

// silu
// Function SILU (same as Swish)
// use activation Silu[x] = x*Sigmoid[x]
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html?highlight=silu#torch.nn.SiLU
ALWI void silu_tile(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_silu<APPROX, SyncHalf>(idst) ));
}

ALWI void silu_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_silu_init<APPROX>() ));
}

/**
 * Pauses the cores so that the debug interface can be used to inspect the value of the registers.
 *
 * Return value: None
 */
ALWI void dbg_halt() {
    PACK (dbg_thread_halt<PackThreadId>());
    UNPACK (dbg_thread_halt<UnpackThreadId>());
    MATH (dbg_thread_halt<MathThreadId>());
}

/**
 * Resumes the execution of the cores after a debug halt.
 *
 * Return value: None
 */
ALWI void dbg_unhalt() {
    PACK (dbg_thread_unhalt<PackThreadId>());
    UNPACK (dbg_thread_unhalt<UnpackThreadId>());
    MATH (dbg_thread_unhalt<MathThreadId>());
}

/**
 * Reads the contents of the specified row of the destination register. It reads 8 dwords at a time.
 *
 * | Argument        | Description                                                                | Type      | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|-----------|-------------------------------------------------------|----------|
 * | row_addr        | The row address in the destination register to read                        | int       |                                                       | True     |
 * | rd_data         | The array of 8 dwords to store the data                                    | uint32_t* |                                                       | True     |
 *
 * Return value: None
*/
ALWI void dbg_read_dest_acc_row(int row_addr, uint32_t *rd_data) {
    MATH (( dbg_get_array_row(dbg_array_id::DEST, row_addr, rd_data)));
}

} // namespace ckernel
