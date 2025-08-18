// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_comp.h"
#include "ckernel_sfpu_unary_comp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
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
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL(unary_ne, RC, APPROX, idst, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_ne_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(unary_ne, APPROX)); }

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
    MATH(SFPU_COMP_INT32_KERNEL(unary_ne, RC, APPROX, idst, param0));
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
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL(unary_eq, RC, APPROX, idst, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_eq_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(unary_eq, APPROX)); }

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
    MATH(SFPU_COMP_INT32_KERNEL(unary_eq, RC, APPROX, idst, param0));
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
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL(unary_gt, RC, APPROX, idst, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_gt_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(unary_gt, APPROX)); }

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
    MATH(SFPU_COMP_INT32_KERNEL_UNDERSCORE(unary_gt, RC, APPROX, idst, param0));
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
ALWI void unary_ge_tile(uint32_t idst, uint32_t param0) { MATH(SFPU_COMP_KERNEL(unary_ge, RC, APPROX, idst, param0)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_ge_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(unary_ge, APPROX)); }

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
    MATH(SFPU_COMP_INT32_KERNEL_UNDERSCORE(unary_ge, RC, APPROX, idst, param0));
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
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL(unary_lt, RC, APPROX, idst, param0));
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
    MATH(SFPU_COMP_INT32_KERNEL_UNDERSCORE(unary_lt, RC, APPROX, idst, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_lt_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(unary_lt, APPROX)); }

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
ALWI void unary_le_tile(uint32_t idst, uint32_t param0) { MATH(SFPU_COMP_KERNEL(unary_le, RC, APPROX, idst, param0)); }

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
    MATH(SFPU_COMP_INT32_KERNEL_UNDERSCORE(unary_le, RC, APPROX, idst, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_le_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(unary_le, APPROX)); }

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
ALWI void gtz_tile(uint32_t idst) { MATH(SFPU_ZERO_KERNEL(greater_than_zero, RC, APPROX, idst)); }

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
ALWI void gtz_tile_int32(uint32_t idst) { MATH(SFPU_ZERO_KERNEL_TYPE(int, greater_than_zero, RC, APPROX, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void gtz_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(greater_than_zero, APPROX)); }

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
ALWI void nez_tile(uint32_t idst) { MATH(SFPU_ZERO_KERNEL(not_equal_zero, RC, APPROX, idst)); }

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
ALWI void nez_tile_int32(uint32_t idst) { MATH(SFPU_ZERO_KERNEL_TYPE(int, not_equal_zero, RC, APPROX, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void nez_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(not_equal_zero, APPROX)); }

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
ALWI void gez_tile(uint32_t idst) { MATH(SFPU_ZERO_KERNEL(greater_than_equal_zero, RC, APPROX, idst)); }

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
ALWI void gez_tile_int32(uint32_t idst) { MATH(SFPU_ZERO_KERNEL_TYPE(int, greater_than_equal_zero, RC, APPROX, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void gez_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(greater_than_equal_zero, APPROX)); }

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
ALWI void ltz_tile(uint32_t idst) { MATH(SFPU_ZERO_KERNEL(less_than_zero, RC, APPROX, idst)); }

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
ALWI void ltz_tile_int32(uint32_t idst) { MATH(SFPU_ZERO_KERNEL_TYPE(int, less_than_zero, RC, APPROX, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void ltz_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(less_than_zero, APPROX)); }

// clang-format off
/**
 * Will store in the output of the compute core True if each element of a tile is equal to zero.
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
ALWI void eqz_tile(uint32_t idst) { MATH(SFPU_ZERO_KERNEL(equal_zero, RC, APPROX, idst)); }

// clang-format off
/**
 * Will store in the output of the compute core True if each element of a tile is equal to zero.
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
ALWI void eqz_tile_int32(uint32_t idst) { MATH(SFPU_ZERO_KERNEL_TYPE(int, equal_zero, RC, APPROX, idst)); }

// clang-format off
/**
 * Will store in the output of the compute core True if each element of a tile is equal to zero.
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
ALWI void eqz_tile_uint16(uint32_t idst) { MATH(SFPU_ZERO_KERNEL_TYPE(uint16, equal_zero, RC, APPROX, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void eqz_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(equal_zero, APPROX)); }

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
ALWI void lez_tile(uint32_t idst) { MATH(SFPU_ZERO_KERNEL(less_than_equal_zero, RC, APPROX, idst)); }

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
ALWI void lez_tile_int32(uint32_t idst) { MATH(SFPU_ZERO_KERNEL_TYPE(int, less_than_equal_zero, RC, APPROX, idst)); }

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
ALWI void nez_tile_uint16(uint32_t idst) { MATH(SFPU_ZERO_KERNEL_TYPE(uint16, not_equal_zero, RC, APPROX, idst)); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void lez_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(less_than_equal_zero, APPROX)); }

}  // namespace ckernel
