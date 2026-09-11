// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logsigmoid.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs logsigmoid in place: logsigmoid(x) = min(x, 0) - log1p(exp(-|x|)).
 *
 * The DST register buffer must be acquired before this call. The output overwrites
 * the input tile at `idst`.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Type     | Valid Range                                           | Required |
 * |----------------|---------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | Index of the input/output tile in DST             | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void logsigmoid_tile(uint32_t idst) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_logsigmoid,
        (APPROX, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC)));
}

/**
 * Initialize logsigmoid operation.
 * Must be called before logsigmoid_tile.
 *
 * Return value: None
 */
ALWI void logsigmoid_tile_init() {
    MATH((SFPU_UNARY_INIT_FN(unused, sfpu::logsigmoid_init, (APPROX, DST_ACCUM_MODE))));
}

}  // namespace ckernel
