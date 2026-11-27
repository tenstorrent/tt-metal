// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binop_with_unary.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#ifndef ARCH_QUASAR
#include "api/compute/legacy_vector_mode.h"
// Every TRISC: the binop_with_scalar_tile signature names llk::sfpu::BinopScalarMode.
#include "sfpu/macro/binop_scalar.h"
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
 * | vector_mode    | The faces to compute (default: VectorMode::RC)                             | VectorMode | Subject to specific hardware/kernel limits                  | False    |
 */
// clang-format on
enum { ADD_UNARY = 0, SUB_UNARY = 1, MUL_UNARY = 2, DIV_UNARY = 3, RSUB_UNARY = 4 };
#ifndef ARCH_QUASAR
// clang-format off
/**
 * Performs element-wise computation of output = input <mode> scalar on the region *shape* of the tiles in DST register.
 * Out of place; pass the same index for output and input to operate in place.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument | Description                                                                 | Type                       | Valid Range                                       | Required |
 * |----------|-----------------------------------------------------------------------------|----------------------------|---------------------------------------------------|----------|
 * | shape    | Region of the tiles to process (template argument)                          | TensorShape                | At most 2x2 faces                                 | True     |
 * | mode     | Add, Sub, Mul, Div (multiplies by the scalar) or Rsub (template argument)   | llk::sfpu::BinopScalarMode | Int32 supports Add, Sub and Rsub only             | True     |
 * | format   | The format DST holds (template argument)                                    | DataFormat                 | Float32, Float16_b or Int32                       | False    |
 * | padding  | Whether the call may compute on lanes outside shape (template argument)     | PaddingAgnostic            | Yes (default) or No                               | False    |
 * | output   | The index of the tile in DST REG the result is written to                   | uint32_t                   | Must be less than the acquired size of DST REG    | True     |
 * | input    | The index of the tile in DST REG read                                       | uint32_t                   | Must be less than the acquired size of DST REG    | True     |
 * | scalar   | The scalar's bits: fp32, or int32 for Int32. For Div, 1 / divisor           | uint32_t                   | Any                                               | True     |
 */
// clang-format on
template <
    TensorShape shape,
    llk::sfpu::BinopScalarMode mode,
    DataFormat format = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b,
    llk::sfpu::PaddingAgnostic padding = llk::sfpu::PaddingAgnostic::Yes>
ALWI void binop_with_scalar_tile(uint32_t output, uint32_t input, uint32_t scalar) {
    MATH((llk::sfpu::BinopScalar::execute<mode, format, DST_SYNC_MODE, shape, padding>(output, input, scalar)));
}
#endif

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_unary_tile(uint32_t idst, uint32_t param1, VectorMode vector_mode = VectorMode::RC) {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst,
        vector_mode,
        param1));
#else
    MATH((detail::with_legacy_vector_mode_shape(vector_mode, [&](auto shape) __attribute__((always_inline)) {
        llk::sfpu::BinopScalar::execute<
            llk::sfpu::BinopScalarMode::Add,
            is_fp32_dest_acc_en ? DataFormat::Float32 : DataFormat::Float16_b,
            DST_SYNC_MODE,
            decltype(shape)::value>(idst, idst, param1);
    })));
#endif
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sub_unary_tile(uint32_t idst, uint32_t param1, VectorMode vector_mode = VectorMode::RC) {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_binop_with_scalar,
        (APPROX, SUB_UNARY, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst,
        vector_mode,
        param1));
#else
    MATH((detail::with_legacy_vector_mode_shape(vector_mode, [&](auto shape) __attribute__((always_inline)) {
        llk::sfpu::BinopScalar::execute<
            llk::sfpu::BinopScalarMode::Sub,
            is_fp32_dest_acc_en ? DataFormat::Float32 : DataFormat::Float16_b,
            DST_SYNC_MODE,
            decltype(shape)::value>(idst, idst, param1);
    })));
#endif
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_unary_tile(uint32_t idst, uint32_t param1, VectorMode vector_mode = VectorMode::RC) {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst,
        vector_mode,
        param1));
#else
    MATH((detail::with_legacy_vector_mode_shape(vector_mode, [&](auto shape) __attribute__((always_inline)) {
        llk::sfpu::BinopScalar::execute<
            llk::sfpu::BinopScalarMode::Mul,
            is_fp32_dest_acc_en ? DataFormat::Float32 : DataFormat::Float16_b,
            DST_SYNC_MODE,
            decltype(shape)::value>(idst, idst, param1);
    })));
#endif
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void div_unary_tile(uint32_t idst, uint32_t param1, VectorMode vector_mode = VectorMode::RC) {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_binop_with_scalar,
        (APPROX, DIV_UNARY, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst,
        vector_mode,
        param1));
#else
    MATH((detail::with_legacy_vector_mode_shape(vector_mode, [&](auto shape) __attribute__((always_inline)) {
        llk::sfpu::BinopScalar::execute<
            llk::sfpu::BinopScalarMode::Div,
            is_fp32_dest_acc_en ? DataFormat::Float32 : DataFormat::Float16_b,
            DST_SYNC_MODE,
            decltype(shape)::value>(idst, idst, param1);
    })));
#endif
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsub_unary_tile(uint32_t idst, uint32_t param1, VectorMode vector_mode = VectorMode::RC) {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_binop_with_scalar,
        (APPROX, RSUB_UNARY, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst,
        vector_mode,
        param1));
#else
    MATH((detail::with_legacy_vector_mode_shape(vector_mode, [&](auto shape) __attribute__((always_inline)) {
        llk::sfpu::BinopScalar::execute<
            llk::sfpu::BinopScalarMode::Rsub,
            is_fp32_dest_acc_en ? DataFormat::Float32 : DataFormat::Float16_b,
            DST_SYNC_MODE,
            decltype(shape)::value>(idst, idst, param1);
    })));
#endif
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

ALWI void add_unary_tile_int32(uint32_t idst, uint32_t param1) {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_add_int32,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param1));
#else
    MATH((llk::sfpu::BinopScalar::
              execute<llk::sfpu::BinopScalarMode::Add, DataFormat::Int32, DST_SYNC_MODE, DEFAULT_TENSOR_SHAPE>(
                  idst, idst, param1)));
#endif
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

ALWI void sub_unary_tile_int32(uint32_t idst, uint32_t param1) {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sub_int32,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param1));
#else
    MATH((llk::sfpu::BinopScalar::
              execute<llk::sfpu::BinopScalarMode::Sub, DataFormat::Int32, DST_SYNC_MODE, DEFAULT_TENSOR_SHAPE>(
                  idst, idst, param1)));
#endif
}

// clang-format off
/**
* Performs element-wise add operation with uint32 scalar, wrapping modulo 2^32. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
* | param1          | uint32 value scalar                                                        | uint32_t |                                                       | True     |
*/
// clang-format on

ALWI void add_unary_tile_uint32(uint32_t idst, uint32_t param1) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_add_uint32,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param1));
}

// clang-format off
/**
* Performs element-wise sub operation with uint32 scalar, wrapping modulo 2^32. The DST
* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
* compute engine.
*
* Return value: None
*
* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
* | param1          | uint32 value scalar                                                        | uint32_t |                                                       | True     |
*/
// clang-format on

ALWI void sub_unary_tile_uint32(uint32_t idst, uint32_t param1) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sub_uint32,
        (APPROX, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        param1));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void binop_with_scalar_tile_init() {
#ifdef ARCH_QUASAR
    MATH(SFPU_UNARY_INIT(unused));
#else
    MATH((llk::sfpu::BinopScalar::init()));
#endif
}

}  // namespace ckernel
