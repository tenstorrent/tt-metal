// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifndef ARCH_QUASAR
#include "sfpu/ckernel_sfpu_comp.h"
#include "ckernel_sfpu_comp.h"
#include "ckernel_sfpu_unary_comp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#else
#include "ckernel_sfpu_comp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#endif

namespace ckernel {

#ifndef ARCH_QUASAR
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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_ne, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_ne_tile_init() { MATH(SFPU_UNARY_INIT(unary_ne)); }

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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_comp_unary_int,
        (APPROX, SfpuType::unary_ne, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_eq, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_eq_tile_init() { MATH(SFPU_UNARY_INIT(unary_eq)); }

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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_comp_unary_int,
        (APPROX, SfpuType::unary_eq, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_gt, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_gt_tile_init() { MATH(SFPU_UNARY_INIT(unary_gt)); }

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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_comp_unary_int_,
        (APPROX, SfpuType::unary_gt, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_ge, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_ge_tile_init() { MATH(SFPU_UNARY_INIT(unary_ge)); }

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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_comp_unary_int_,
        (APPROX, SfpuType::unary_ge, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_lt, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_comp_unary_int_,
        (APPROX, SfpuType::unary_lt, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_lt_tile_init() { MATH(SFPU_UNARY_INIT(unary_lt)); }

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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_le, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, param0));
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
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_comp_unary_int_,
        (APPROX, SfpuType::unary_le, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_le_tile_init() { MATH(SFPU_UNARY_INIT(unary_le)); }
#endif  // !ARCH_QUASAR

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
ALWI void gtz_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_comp, (APPROX, SfpuType::greater_than_zero), idst, VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_zero_comp,
        (APPROX, DataFormat::Float32, SfpuType::greater_than_zero, SFPU_ITERATIONS),
        idst,
        VectorMode::RC));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void gtz_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(greater_than_zero));
#else
    MATH(SFPU_UNARY_INIT(greater_than_zero, sfpu::init_zero_comp));
#endif
}

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
ALWI void nez_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_comp, (APPROX, SfpuType::not_equal_zero), idst, VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_zero_comp,
        (APPROX, DataFormat::Float32, SfpuType::not_equal_zero, SFPU_ITERATIONS),
        idst,
        VectorMode::RC));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void nez_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(not_equal_zero));
#else
    MATH(SFPU_UNARY_INIT(not_equal_zero, sfpu::init_zero_comp));
#endif
}

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
ALWI void gez_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_comp,
        (APPROX, SfpuType::greater_than_equal_zero),
        idst,
        VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_zero_comp,
        (APPROX, DataFormat::Float32, SfpuType::greater_than_equal_zero, SFPU_ITERATIONS),
        idst,
        VectorMode::RC));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void gez_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(greater_than_equal_zero));
#else
    MATH(SFPU_UNARY_INIT(greater_than_equal_zero, sfpu::init_zero_comp));
#endif
}

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
ALWI void ltz_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_comp, (APPROX, SfpuType::less_than_zero), idst, VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_zero_comp,
        (APPROX, DataFormat::Float32, SfpuType::less_than_zero, SFPU_ITERATIONS),
        idst,
        VectorMode::RC));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void ltz_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(less_than_zero));
#else
    MATH(SFPU_UNARY_INIT(less_than_zero, sfpu::init_zero_comp));
#endif
}

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
ALWI void eqz_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_comp, (APPROX, SfpuType::equal_zero), idst, VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_zero_comp,
        (APPROX, DataFormat::Float32, SfpuType::equal_zero, SFPU_ITERATIONS),
        idst,
        VectorMode::RC));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void eqz_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(equal_zero));
#else
    MATH(SFPU_UNARY_INIT(equal_zero, sfpu::init_zero_comp));
#endif
}

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
ALWI void lez_tile(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_comp, (APPROX, SfpuType::less_than_equal_zero), idst, VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_zero_comp,
        (APPROX, DataFormat::Float32, SfpuType::less_than_equal_zero, SFPU_ITERATIONS),
        idst,
        VectorMode::RC));
#endif
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void lez_tile_init() {
#ifndef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(less_than_equal_zero));
#else
    MATH(SFPU_UNARY_INIT(less_than_equal_zero, sfpu::init_zero_comp));
#endif
}

// Integer comparison-to-zero variants. These read int32/uint operands from Dest, which on Quasar
// requires 32-bit unpack-to-Dest that is not supported yet, so the whole block stays gated off there.
#ifndef ARCH_QUASAR
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
ALWI void gtz_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_comp_int,
        (APPROX, SfpuType::greater_than_zero),
        idst,
        VectorMode::RC));
}

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
ALWI void nez_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_comp_int, (APPROX, SfpuType::not_equal_zero), idst, VectorMode::RC));
}

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
ALWI void gez_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_comp_int,
        (APPROX, SfpuType::greater_than_equal_zero),
        idst,
        VectorMode::RC));
}

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
ALWI void ltz_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_comp_int, (APPROX, SfpuType::less_than_zero), idst, VectorMode::RC));
}

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
ALWI void eqz_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_comp_int, (APPROX, SfpuType::equal_zero), idst, VectorMode::RC));
}

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
ALWI void eqz_tile_uint16(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_comp_uint16, (APPROX, SfpuType::equal_zero), idst, VectorMode::RC));
}

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
ALWI void eqz_tile_uint32(uint32_t idst) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_eqz_uint32, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC)));
}

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
ALWI void lez_tile_int32(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_comp_int,
        (APPROX, SfpuType::less_than_equal_zero),
        idst,
        VectorMode::RC));
}

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
ALWI void nez_tile_uint16(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_comp_uint16,
        (APPROX, SfpuType::not_equal_zero),
        idst,
        VectorMode::RC));
}

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
ALWI void nez_tile_uint32(uint32_t idst) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_nez_uint32, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC)));
}
#endif  // !ARCH_QUASAR

}  // namespace ckernel
