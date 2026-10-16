// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise xlogy operation y = xlogy(x0, x1) with x0 as first operand and x1 as second operand.
 * Output overwrites first operand in DST.
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument              | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void xlogy_binary_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((sfpu::BinaryFloat<APPROX, BinaryOp::XLOGY, is_fp32_dest_acc_en>::run(idst0, idst1, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void xlogy_binary_tile_init() { MATH((sfpu::BinaryFloat<APPROX, BinaryOp::XLOGY>::init())); }

}  // namespace ckernel
