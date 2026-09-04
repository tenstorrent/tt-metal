// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_comp.h"
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_unary_comp.h"
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_ne_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Ne, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_ne_tile_init() {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Ne, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_ne_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::UnaryCompMode::Ne, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_eq_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Eq, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_eq_tile_init() {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Eq, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_eq_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::UnaryCompMode::Eq, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_gt_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Gt, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_gt_tile_init() {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Gt, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_gt_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::UnaryCompMode::Gt, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_ge_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Ge, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_ge_tile_init() {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Ge, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_ge_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::UnaryCompMode::Ge, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_lt_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Lt, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_lt_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::UnaryCompMode::Lt, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_lt_tile_init() {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Lt, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_le_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Le, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_le_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::UnaryCompMode::Le, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_le_tile_init() {
    MATH((sfpu::UnaryComp<APPROX, sfpu::UnaryCompMode::Le, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gtz_tile(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::GtZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gtz_tile_init() {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::GtZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nez_tile(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::NeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nez_tile_init() {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::NeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gez_tile(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::GeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gez_tile_init() {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::GeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ltz_tile(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::LtZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ltz_tile_init() {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::LtZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void eqz_tile(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::EqZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void eqz_tile_init() {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::EqZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lez_tile(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::LeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lez_tile_init() {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Float16_b, sfpu::ZeroCompMode::LeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gtz_tile_int32(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Int32, sfpu::ZeroCompMode::GtZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nez_tile_int32(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Int32, sfpu::ZeroCompMode::NeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void gez_tile_int32(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Int32, sfpu::ZeroCompMode::GeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ltz_tile_int32(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Int32, sfpu::ZeroCompMode::LtZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void eqz_tile_int32(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Int32, sfpu::ZeroCompMode::EqZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void eqz_tile_uint16(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::UInt16, sfpu::ZeroCompMode::EqZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void eqz_tile_uint32(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::UInt32, sfpu::ZeroCompMode::EqZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lez_tile_int32(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::Int32, sfpu::ZeroCompMode::LeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nez_tile_uint16(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::UInt16, sfpu::ZeroCompMode::NeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nez_tile_uint32(uint32_t idst) {
    MATH((sfpu::ZeroComp<APPROX, DataFormat::UInt32, sfpu::ZeroCompMode::NeZ, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC)));
}
#endif  // !ARCH_QUASAR

}  // namespace ckernel
