// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_softplus.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of softplus (`1/beta * log(1 + exp(beta * x))`) on each element
 * of a tile in DST register at index tile_index. Any input value greater than the provided threshold
 * with return itself. The DST register buffer must be in acquired state via *acquire_dst* call. This
 * call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index      | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | beta            | Beta used in softplus calculation                                          | uint32_t | Greater than 0                                        | True     |
 * | beta_reciprocal | Reciprocal of beta (1/beta) used in softplus calculation                   | uint32_t | Greater than 0                                        | True     |
 * | threshold       | Threshold used in softplus calculation                                     | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
ALWI void softplus_tile(uint32_t idst, uint32_t beta, uint32_t beta_reciprocal, uint32_t threshold) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_softplus,
        (APPROX, DST_ACCUM_MODE),
        idst,
        VectorMode::RC,
        beta,
        beta_reciprocal,
        threshold));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softplus_tile_init() { MATH(SFPU_UNARY_INIT(softplus)); }

#ifndef ARCH_QUASAR
// Pack-thread variants: Quasar has no pack-thread SFPU, so these are gated off there.
ALWI void softplus_tile_pack(uint32_t idst, uint32_t beta, uint32_t beta_reciprocal, uint32_t threshold) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_softplus,
        (APPROX, DST_ACCUM_MODE),
        idst,
        VectorMode::RC,
        beta,
        beta_reciprocal,
        threshold));
}

ALWI void softplus_tile_init_pack() { PACK(SFPU_UNARY_INIT(softplus)); }
#endif  // !ARCH_QUASAR

}  // namespace ckernel
