// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_erf.h"
#include "ckernel_sfpu_erfc.h"
#endif
#endif

namespace ckernel {
#ifndef ARCH_QUASAR
/************** ERF *****************/
/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = true, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void erf_tile_init() {
    MATH((sfpu::Erf<fast_and_approx, false /*IS_ERFC*/, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

// clang-format off
/**
 * Performs element-wise computation of error function on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool fast_and_approx = true, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void erf_tile(uint32_t idst) {
    MATH((sfpu::Erf<fast_and_approx, false /*IS_ERFC*/, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

/************** ERFC *****************/

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void erfc_tile_init() {
    MATH((sfpu::Erf<true /*APPROXIMATION_MODE*/, true /*IS_ERFC*/, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

// clang-format off
/**
 * Performs element-wise computation of complimentary error function on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void erfc_tile(uint32_t idst) {
    MATH((sfpu::Erf<true /*APPROXIMATION_MODE*/, true /*IS_ERFC*/, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

#endif

}  // namespace ckernel
