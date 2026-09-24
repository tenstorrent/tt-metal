// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_comp.h"
#ifndef ARCH_QUASAR
// The scalar compares do not exist on Quasar.
#include "ckernel_sfpu_unary_comp.h"
#endif
#endif

namespace ckernel {

// The scalar compares (float and int32) do not exist on Quasar.
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
ALWI void unary_ne_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::ne>::run(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_ne_tile_init() { MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::ne>::init())); }

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
ALWI void unary_ne_tile_int32(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::CompareOp::ne>::run(idst, param0)));
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
ALWI void unary_eq_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::eq>::run(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_eq_tile_init() { MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::eq>::init())); }

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
ALWI void unary_eq_tile_int32(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::CompareOp::eq>::run(idst, param0)));
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
ALWI void unary_gt_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::gt>::run(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_gt_tile_init() { MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::gt>::init())); }

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
ALWI void unary_gt_tile_int32(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::CompareOp::gt>::run(idst, param0)));
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
ALWI void unary_ge_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::ge>::run(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_ge_tile_init() { MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::ge>::init())); }

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
ALWI void unary_ge_tile_int32(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::CompareOp::ge>::run(idst, param0)));
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
ALWI void unary_lt_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::lt>::run(idst, param0)));
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
ALWI void unary_lt_tile_int32(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::CompareOp::lt>::run(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_lt_tile_init() { MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::lt>::init())); }

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
ALWI void unary_le_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::le>::run(idst, param0)));
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
ALWI void unary_le_tile_int32(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::UnaryCompInt<APPROX, sfpu::CompareOp::le>::run(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_le_tile_init() { MATH((sfpu::UnaryComp<APPROX, sfpu::CompareOp::le>::init())); }
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
ALWI void gtz_tile(std::uint32_t idst) { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::gt>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void gtz_tile_init() { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::gt>::init())); }

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
ALWI void nez_tile(std::uint32_t idst) { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::ne>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void nez_tile_init() { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::ne>::init())); }

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
ALWI void gez_tile(std::uint32_t idst) { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::ge>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void gez_tile_init() { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::ge>::init())); }

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
ALWI void ltz_tile(std::uint32_t idst) { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::lt>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void ltz_tile_init() { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::lt>::init())); }

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
ALWI void eqz_tile(std::uint32_t idst) { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::eq>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void eqz_tile_init() { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::eq>::init())); }

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
ALWI void lez_tile(std::uint32_t idst) { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::le>::run(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void lez_tile_init() { MATH((sfpu::ZeroComp<APPROX, sfpu::CompareOp::le>::init())); }

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
ALWI void gtz_tile_int32(std::uint32_t idst) {
    MATH((sfpu::ZeroCompInt<APPROX, sfpu::CompareOp::gt, DataFormat::Int32>::run(idst)));
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
ALWI void nez_tile_int32(std::uint32_t idst) {
    MATH((sfpu::ZeroCompInt<APPROX, sfpu::CompareOp::ne, DataFormat::Int32>::run(idst)));
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
ALWI void gez_tile_int32(std::uint32_t idst) {
    MATH((sfpu::ZeroCompInt<APPROX, sfpu::CompareOp::ge, DataFormat::Int32>::run(idst)));
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
ALWI void ltz_tile_int32(std::uint32_t idst) {
    MATH((sfpu::ZeroCompInt<APPROX, sfpu::CompareOp::lt, DataFormat::Int32>::run(idst)));
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
ALWI void eqz_tile_int32(std::uint32_t idst) {
    MATH((sfpu::ZeroCompInt<APPROX, sfpu::CompareOp::eq, DataFormat::Int32>::run(idst)));
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
ALWI void eqz_tile_uint16(std::uint32_t idst) {
    MATH((sfpu::ZeroCompInt<APPROX, sfpu::CompareOp::eq, DataFormat::UInt16>::run(idst)));
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
ALWI void eqz_tile_uint32(std::uint32_t idst) {
    MATH((sfpu::ZeroCompInt<APPROX, sfpu::CompareOp::eq, DataFormat::UInt32>::run(idst)));
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
ALWI void lez_tile_int32(std::uint32_t idst) {
    MATH((sfpu::ZeroCompInt<APPROX, sfpu::CompareOp::le, DataFormat::Int32>::run(idst)));
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
ALWI void nez_tile_uint16(std::uint32_t idst) {
    MATH((sfpu::ZeroCompInt<APPROX, sfpu::CompareOp::ne, DataFormat::UInt16>::run(idst)));
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
ALWI void nez_tile_uint32(std::uint32_t idst) {
    MATH((sfpu::ZeroCompInt<APPROX, sfpu::CompareOp::ne, DataFormat::UInt32>::run(idst)));
}
#endif  // !ARCH_QUASAR

}  // namespace ckernel
