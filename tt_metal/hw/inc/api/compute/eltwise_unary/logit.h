// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logit.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = false>
ALWI void logit_tile_init() {
    MATH(SFPU_THREE_TEMPLATE_PARAM_INIT(logit, sfpu::logit_init, APPROX, fast_and_approx, DST_ACCUM_MODE));
}

// clang-format off
/**
 * Performs element-wise computation of logit, logit(x) = log(x / (1 - x)), on
 * each element of a tile in DST register at index tile_index. The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is
 * blocking and is only available on the compute engine.
 *
 * Intended for the fp32 dest accumulation path; it fuses the hybrid
 * log1p / log formulation (see ckernel_sfpu_logit.h) into a single SFPU pass.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool fast_and_approx = false>
ALWI void logit_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_LOG1P_FN(calculate_logit, RC, APPROX, fast_and_approx, DST_ACCUM_MODE, idst));
}

}  // namespace ckernel
