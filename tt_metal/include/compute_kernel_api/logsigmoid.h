// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_logsigmoid.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs logsigmoid operation: logsigmoid(x) = -softplus(-x) = -log(1 + exp(-x))
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Type     | Valid Range                                           | Required |
 * |----------------|---------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_in0       | Index of tile in DST with input (x)               | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_in1       | Index of tile in DST with exp(-x)                 | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_out       | Index of tile in DST for output                   | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void logsigmoid_tile(uint32_t idst_in0, uint32_t idst_in1, uint32_t idst_out) {
    MATH((llk_math_eltwise_binary_sfpu_logsigmoid<APPROX>(idst_in0, idst_in1, idst_out)));
}

/**
 * Initialize logsigmoid operation.
 * Must be called before logsigmoid_tile.
 *
 * Return value: None
 */
ALWI void logsigmoid_tile_init() { MATH((llk_math_eltwise_binary_sfpu_logsigmoid_init<APPROX>())); }

}  // namespace ckernel
