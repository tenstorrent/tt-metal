// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_rsqrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// Do not reinterpret a removed legacy_compat Boolean argument as a fast-mode or DEST override.
enum class RsqrtMode { Default, Fast };

/**
 * Please refer to documentation for any_init.
 */
ALWI void rsqrt_tile_init() { MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::rsqrt_init, (APPROX))); }

// clang-format off
/**
 * Performs element-wise computation of reciprocal sqrt on each element of a
 * tile in DST register at index idst. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 * The optional fast mode is selected with RsqrtMode::Fast; RsqrtMode::Default preserves
 * the kernel's normal approximation behavior. Legacy Boolean template arguments are not accepted.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <RsqrtMode mode = RsqrtMode::Default, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsqrt_tile(uint32_t idst) {
    [[maybe_unused]] constexpr bool FAST_APPROX = mode == RsqrtMode::Fast;
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_rsqrt,
        (APPROX, 8 /* ITERATIONS */, is_fp32_dest_acc_en, FAST_APPROX),
        idst,
        VectorMode::RC));
}

}  // namespace ckernel
