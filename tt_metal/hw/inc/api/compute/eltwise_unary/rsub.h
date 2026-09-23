// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binop_with_unary.h"
#include "ckernel_sfpu_rsub_int32.h"
#endif

namespace ckernel {
// RSUB : rsub(x,y) = y-x

// clang-format off
/**
 * Performs element-wise computation of rsub ( rsub(x,y) = y -x) on each element of a tile and y is a constant param
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | scalar         | Constant value that is being subtracted from                               | uint32_t |                                                       | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsub_tile(uint32_t idst, uint32_t scalar) {
    MATH((
        sfpu::BinopWithScalar<APPROX, RSUB_UNARY, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
            idst, VectorMode::RC, scalar)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsub_tile_init() {
    MATH(
        (sfpu::BinopWithScalar<APPROX, RSUB_UNARY, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

// clang-format off
/**
 * Performs element-wise computation of rsub ( rsub(x,y) = y -x) on each element of a tile and y is a constant param for int32 dtype
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | scalar         | Constant value that is being subtracted from                               | uint32_t |                                                       | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsub_unary_int32_tile(uint32_t idst, uint32_t scalar) {
    MATH((sfpu::RsubScalarInt32<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC, scalar)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsub_unary_int32_tile_init() {
    MATH((sfpu::RsubScalarInt32<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
