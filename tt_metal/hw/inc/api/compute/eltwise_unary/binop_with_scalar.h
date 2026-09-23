// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binop_with_unary.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs a simple elementwise binop with scalar operation on the input: y = binop(x,scalar)
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                                 | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform requested operation| uint32_t | Must be less than the size of the DST register buffer       | True     |
 * | mode           | 0, 1, 2, 3, and 4                                                          | uint32_t | 0, 1, 2, 3, 4 corresponding to add, mul, sub, div, and rsub | True     |
 * | param1         | fp32 value scalar encoded as uint32                                        | uint32_t | Must be less than the size of the DST register buffer       | True     |
 */
// clang-format on
enum { ADD_UNARY = 0, SUB_UNARY = 1, MUL_UNARY = 2, DIV_UNARY = 3, RSUB_UNARY = 4 };
#ifndef ARCH_QUASAR
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_unary_tile(uint32_t idst, uint32_t param1) {
    MATH(
        (sfpu::BinopWithScalar<APPROX, ADD_UNARY, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst, VectorMode::RC, param1)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sub_unary_tile(uint32_t idst, uint32_t param1) {
    MATH(
        (sfpu::BinopWithScalar<APPROX, SUB_UNARY, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst, VectorMode::RC, param1)));
}
#endif

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_unary_tile(uint32_t idst, uint32_t param1) {
    MATH(
        (sfpu::BinopWithScalar<APPROX, MUL_UNARY, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst, VectorMode::RC, param1)));
}

#ifndef ARCH_QUASAR
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void div_unary_tile(uint32_t idst, uint32_t param1) {
    MATH(
        (sfpu::BinopWithScalar<APPROX, DIV_UNARY, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst, VectorMode::RC, param1)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsub_unary_tile(uint32_t idst, uint32_t param1) {
    MATH((
        sfpu::BinopWithScalar<APPROX, RSUB_UNARY, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst, VectorMode::RC, param1)));
}

// clang-format off
/**
* Performs element-wise add operation with int32 scalar. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
* | param1          | int32 value scalar encoded as uint32                                       | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_unary_tile_int32(uint32_t idst, uint32_t param1) {
    MATH((sfpu::BinopWithScalar<APPROX, ADD_UNARY, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param1)));
}

// clang-format off
/**
* Performs element-wise sub operation with int32 scalar. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
* | param1          | int32 value scalar encoded as uint32                                       | uint32_t | Must be less than the size of the DST register buffer | True     |
*/
// clang-format on

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sub_unary_tile_int32(uint32_t idst, uint32_t param1) {
    MATH((sfpu::BinopWithScalar<APPROX, SUB_UNARY, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param1)));
}
#endif

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void binop_with_scalar_tile_init() {
    MATH((sfpu::BinopWithScalar<APPROX, MUL_UNARY, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
