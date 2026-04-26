// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary.h"
#include "ckernel_sfpu_binary_pow.h"
#include "ckernel_sfpu_binary_comp.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise binop operation with the two floating point inputs: y = binop(x0,x1)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 8, false),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, 8, false),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_div,
        (APPROX, ckernel::BinaryOp::DIV, 8, DST_ACCUM_MODE),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void rsub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::RSUB, 8, false),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void power_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_pow,
        (APPROX, 8, DST_ACCUM_MODE),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void eq_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8, SfpuType::eq),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void ne_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8, SfpuType::ne),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void lt_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8, SfpuType::lt),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void gt_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8, SfpuType::gt),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void le_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8, SfpuType::le),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

ALWI void ge_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binary_comp_fp32,
        (APPROX, 8, SfpuType::ge),
        idst0,
        idst1,
        odst,
        (int)VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void add_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_CB(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::ADD))));
}

ALWI void sub_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_CB(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::SUB))));
}

ALWI void mul_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_CB(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
}

ALWI void div_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_CB(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::DIV))));
}

ALWI void rsub_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_CB(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::RSUB))));
}

ALWI void power_binary_tile_init() { MATH((SFPU_BINARY_INIT_CB(unused, sfpu::sfpu_binary_pow_init, (APPROX)))); }

ALWI void eq_binary_tile_init() { MATH((SFPU_BINARY_INIT(eq))); }

ALWI void ne_binary_tile_init() { MATH((SFPU_BINARY_INIT(ne))); }

ALWI void lt_binary_tile_init() { MATH((SFPU_BINARY_INIT(lt))); }

ALWI void gt_binary_tile_init() { MATH((SFPU_BINARY_INIT(gt))); }

ALWI void le_binary_tile_init() { MATH((SFPU_BINARY_INIT(le))); }

ALWI void ge_binary_tile_init() { MATH((SFPU_BINARY_INIT(ge))); }

}  // namespace ckernel
