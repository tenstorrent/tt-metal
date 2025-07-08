// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_comp.h"
#include "llk_math_eltwise_unary_sfpu_unary_comp.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// unary ne : if x != value --> 1.0, else 0.0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1.0 if x!=value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_ne_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_ne<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_ne_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_ne_init<APPROX>())); }

// unary ne : if x != value --> 1, else 0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1 if x!=value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_ne_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_ne_int32<APPROX>(idst, param0)));
}

// unary eq : if x == value --> 1.0, else 0.0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1.0 if x==value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_eq_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_eq<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_eq_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_eq_init<APPROX>())); }

// unary eq : if x == value --> 1, else 0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1 if x==value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_eq_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_eq_int32<APPROX>(idst, param0)));
}

// unary gt : if x > value --> 1.0, else 0.0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1 if x > value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_gt_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_gt<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_gt_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_gt_init<APPROX>())); }

// unary gt : if x > value --> 1, else 0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1 if x>value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_gt_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_gt_int32<APPROX>(idst, param0)));
}

// unary ge : if x >= value --> 1.0, else 0.0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1 if x >= value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_ge_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_ge<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_ge_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_ge_init<APPROX>())); }

// unary ge : if x >= value --> 1, else 0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1 if x>value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_ge_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_ge_int32<APPROX>(idst, param0)));
}

// unary lt : if x < value --> 1.0, else 0.0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1 if x < value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_lt_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_lt<APPROX>(idst, param0)));
}

// unary lt : if x < value --> 1, else 0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1 if x<value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_lt_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_lt_int32<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_lt_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_lt_init<APPROX>())); }

// unary le : if x <= value --> 1.0, else 0.0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1 if x <= value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_le_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_le<APPROX>(idst, param0)));
}

// unary le : if x <= value --> 1, else 0
// clang-format off
/**
 * Performs element-wise computation of:  result = 1 if x<value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The value to be compared with the input tensor                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void unary_le_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_le_int32<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_le_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_le_init<APPROX>())); }

// clang-format off
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
// clang-format on
ALWI void gtz_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_gtz<APPROX>(idst))); }

// clang-format off
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
// clang-format on
ALWI void gtz_tile_int32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_gtz_int32<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void gtz_tile_init() { MATH((llk_math_eltwise_unary_sfpu_gtz_init<APPROX>())); }

// clang-format off
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
// clang-format on
ALWI void nez_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_nez<APPROX>(idst))); }

// clang-format off
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
// clang-format on
ALWI void nez_tile_int32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_nez_int32<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void nez_tile_init() { MATH((llk_math_eltwise_unary_sfpu_nez_init<APPROX>())); }

// clang-format off
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
// clang-format on
ALWI void gez_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_gez<APPROX>(idst))); }

// clang-format off
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
// clang-format on
ALWI void gez_tile_int32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_gez_int32<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void gez_tile_init() { MATH((llk_math_eltwise_unary_sfpu_gez_init<APPROX>())); }

// clang-format off
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
// clang-format on
ALWI void ltz_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_ltz<APPROX>(idst))); }

// clang-format off
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
// clang-format on
ALWI void ltz_tile_int32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_ltz_int32<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void ltz_tile_init() { MATH((llk_math_eltwise_unary_sfpu_ltz_init<APPROX>())); }

// clang-format off
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
// clang-format on
ALWI void eqz_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_eqz<APPROX>(idst))); }

// clang-format off
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
// clang-format on
ALWI void eqz_tile_int32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_eqz_int32<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void eqz_tile_init() { MATH((llk_math_eltwise_unary_sfpu_eqz_init<APPROX>())); }

// clang-format off
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
// clang-format on
ALWI void lez_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_lez<APPROX>(idst))); }

// clang-format off
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
// clang-format on
ALWI void lez_tile_int32(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_lez_int32<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void lez_tile_init() { MATH((llk_math_eltwise_unary_sfpu_lez_init<APPROX>())); }

}  // namespace ckernel
