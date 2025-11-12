// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_logsigmoid.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Performs logsigmoid operation using pre-computed scaled input and exponential values.
 *
 * Return value: None
 *
 * | Argument       | Description                                                              | Type     | Valid Range
 * | Required |
 * |----------------|--------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_in0       | Index of tile in DST with scaled input (beta * x)                        | uint32_t | Must be less
 * than the size of the DST register buffer | True     | | idst_in1       | Index of tile in DST with exp value
 * exp(-beta * x)                       | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_out       | Index of tile in DST for output                                          | uint32_t | Must be less
 * than the size of the DST register buffer | True     | | param0         | Beta value encoded as uint32_t | uint32_t |
 * Float value bit-cast to uint32_t                      | True     | | param1         | Threshold value encoded as
 * uint32_t                                      | uint32_t | Float value bit-cast to uint32_t                      |
 * True     |
 */
ALWI void logsigmoid_tile(uint32_t idst_in0, uint32_t idst_in1, uint32_t idst_out, uint32_t param0, uint32_t param1) {
#ifdef TRISC_MATH
    MATH((llk_math_eltwise_unary_sfpu_logsigmoid<APPROX>(idst_in0, idst_in1, idst_out, param0, param1)));
#endif
}

/**
 * Initialize logsigmoid operation.
 * Must be called before logsigmoid_tile.
 *
 * Return value: None
 */
ALWI void logsigmoid_tile_init() {
#ifdef TRISC_MATH
    MATH((llk_math_eltwise_unary_sfpu_logsigmoid_init<APPROX>()));
#endif
}

}  // namespace ckernel
