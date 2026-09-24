// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_rpow.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void rpow_tile_init() { MATH((sfpu::Rpow<APPROX>::init())); }
// clang-format off
/**
 * Performs element-wise computation of the rpow on each element of a tile
 * where rpow(exponent, scalar_base) = pow(scalar_base, exponent)
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | base_val       | The base value to raise to the power of each element in the tile            | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rpow_tile(std::uint32_t idst, std::uint32_t base_val) {
    MATH((sfpu::Rpow<APPROX, 8 /* ITERATIONS */, is_fp32_dest_acc_en>::run(idst, base_val)));
}

/**
 * Legacy overload selecting the faces to process with a VectorMode. Prefer the overload above, which
 * processes the full tile.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rpow_tile(std::uint32_t idst, std::uint32_t base_val, VectorMode vector_mode) {
    MATH((sfpu::Rpow<APPROX, 8 /* ITERATIONS */, is_fp32_dest_acc_en>::run_vector_mode(vector_mode, idst, base_val)));
}

}  // namespace ckernel
