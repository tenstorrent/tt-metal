// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_remainder.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise remainder computation on input x by y , where x is each element of a tile
 * in DST register at index tile_index. The input can be of float data type. The denominator is provided to
 * remainder_tile_init and loaded into the SFPU constant registers. The
 * DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on
 * the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform remainder operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void remainder_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_remainder, (APPROX), idst, VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void remainder_tile_init(uint32_t param0, uint32_t param1) {
    MATH(SFPU_UNARY_INIT_FN_ARGS(remainder, sfpu::init_remainder, (APPROX), param0, param1));
}

// clang-format off
/**
 * Performs element-wise unsigned remainder computation on input x by the uint32 scalar divisor,
 * where x is each element of a tile in DST register at index tile_index. The result is x mod divisor
 * in [0, divisor). The DST register buffer must be in acquired state via *acquire_dst* call. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform remainder operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0         | The unsigned divisor                                                        | uint32_t | [1, 4294967295]                                       | True     |
 */
// clang-format on
ALWI void remainder_tile_uint32(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_remainder_uint32_scalar,
        (APPROX /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/),
        idst,
        VectorMode::RC,
        param0 /*divisor*/));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void remainder_tile_uint32_init() {
    MATH(SFPU_UNARY_INIT_FN(remainder_uint32, sfpu::remainder_uint32_init, (APPROX)));
}

}  // namespace ckernel
