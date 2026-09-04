// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_lcm.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise LCM operation on two inputs: y = lcm(x0, x1).
 * Both inputs must be int32 with values constrained to |value| ≤ 2^15-1 (32,767).
 * Output overwrites odst in DST.
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lcm_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::Lcm<DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst0, idst1, odst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lcm_tile_init() {
    MATH((sfpu::Lcm<DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
