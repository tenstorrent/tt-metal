// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_rsqrt.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool legacy_compat = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsqrt_tile_init() {
    MATH((sfpu::Rsqrt<APPROX, false /*FAST_APPROX*/, legacy_compat, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

// clang-format off
/**
 * Performs element-wise computation of reciprocal sqrt on each element of a
 * tile in DST register at index idst. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool legacy_compat = false, bool FAST_APPROX = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rsqrt_tile(uint32_t idst) {
    MATH((sfpu::Rsqrt<APPROX, FAST_APPROX, legacy_compat, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

}  // namespace ckernel
