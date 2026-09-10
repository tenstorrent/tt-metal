// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the element-wise IEEE nextafter operation y = nextafter(x0, x1): the representable value
 * adjacent to x0 in the direction of x1, or x0 itself when the two are equal. The step is one ULP at
 * x0's own magnitude, so the float32 and bfloat16 destinations need separate entry points.
 * Output overwrites first operand in DST.
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument              | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand   | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output          | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nextafter_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary,
        (APPROX, BinaryOp::NEXTAFTER, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void nextafter_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, BinaryOp::NEXTAFTER))));
}

// clang-format off
/**
 * bfloat16 destination variant of nextafter_binary_tile. A bfloat16 tile keeps its mantissa in the
 * top 16 bits of the fp32 dest register, so one of its ULPs is a step of 0x10000 there; stepping by
 * a single fp32 ULP instead would be rounded away when the tile is packed and the value would not
 * move at all.
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nextafter_bf16_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary,
        (APPROX, BinaryOp::NEXTAFTER_BF16, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void nextafter_bf16_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, BinaryOp::NEXTAFTER_BF16))));
}

}  // namespace ckernel
