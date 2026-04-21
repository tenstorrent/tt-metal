// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logsigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
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
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_logsigmoid,
        (APPROX, 8),
        idst_in0,
        idst_in1,
        idst_out,
        (int)VectorMode::RC)));
}

/**
 * Initialize logsigmoid operation.
 * Must be called before logsigmoid_tile.
 *
 * Return value: None
 */
ALWI void logsigmoid_tile_init() { MATH((SFPU_BINARY_INIT(unused))); }

}  // namespace ckernel
