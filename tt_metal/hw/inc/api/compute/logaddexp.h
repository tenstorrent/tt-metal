// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logaddexp.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise log-sum-exp on two inputs: y = log(exp(x0) + exp(x1)),
 * evaluated as max(x0, x1) + log1p(exp(-|x0 - x1|)) so that no intermediate leaves
 * the representable range. Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
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
ALWI void logaddexp_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_logaddexp,
        (APPROX, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void logaddexp_binary_tile_init() {
    // Baseline (add1) binary addrmod setup, like fmod/remainder: the op-specific work
    // is the program-constant load in calculate_sfpu_logaddexp_init. There is no
    // SfpuType::logaddexp — naming one here fails to compile the first time a kernel
    // instantiates this init.
    MATH((SFPU_BINARY_INIT_FN(add1, sfpu::calculate_sfpu_logaddexp_init, (DST_ACCUM_MODE))));
}

}  // namespace ckernel
