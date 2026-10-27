// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_hardmish.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of hardmish(x) = x * clamp(x + 2, 0, 2) / 2
 * (equivalently, x * clamp(0.5 * x + 1, 0, 1)) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * For finite x, the piecewise form is:
 *   x <= -2  =>  0           (scale clamped to 0)
 *   x >= 0   =>  x           (scale clamped to 1)
 *   else     =>  x*(x+2)/2   (quadratic)
 *
 * Non-finite inputs follow IEEE 754 semantics; in particular, x = -inf yields NaN.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 *
 * | Template Parameter | Description                                                                   | Type       | Valid Range                  | Default    |
 * |--------------------|-------------------------------------------------------------------------------|------------|------------------------------|------------|
 * | DEST_16BIT_FORMAT  | Format a 16-bit DST holds; Float16_b results are rounded to nearest-even      | DataFormat | Float16_b, Float16           | Float16_b  |
 */
// clang-format on
template <DataFormat DEST_16BIT_FORMAT = DataFormat::Float16_b>
ALWI void hardmish_tile(std::uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        hardmish,
        (APPROX, DST_ACCUM_MODE, DEST_16BIT_FORMAT, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardmish_tile_init() { MATH(SFPU_UNARY_INIT(hardmish)); }

}  // namespace ckernel
