// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_rdiv.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
ALWI void rdiv_tile_init() { MATH((sfpu::Rdiv<APPROX>::init())); }

// clang-format off
/**
 * Performs element-wise computation of reciprocal divide on each element of a
 * tile in DST register at index idst. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | value          | The numerator value to divide by each element of the tile                  | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <RoundingMode rounding_mode = RoundingMode::None, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rdiv_tile(std::uint32_t dst_index, std::uint32_t value) {
    MATH((sfpu::Rdiv<APPROX, is_fp32_dest_acc_en, rounding_mode, 8 /* ITERATIONS */>::run(dst_index, value)));
}

/**
 * Legacy overload selecting the faces to process with a VectorMode. Prefer the overload above, which
 * processes the full tile.
 */
template <RoundingMode rounding_mode = RoundingMode::None, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void rdiv_tile(std::uint32_t dst_index, std::uint32_t value, VectorMode vector_mode) {
    MATH((sfpu::Rdiv<APPROX, is_fp32_dest_acc_en, rounding_mode, 8 /* ITERATIONS */>::run_vector_mode(
        vector_mode, dst_index, value)));
}

}  // namespace ckernel
