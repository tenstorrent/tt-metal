/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "api/compute/common_globals.h"
#include "tensor_shape.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_negative.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

#ifdef ARCH_QUASAR
ALWI void negative_tile_init() { MATH(SFPU_UNARY_INIT(negative)); }

ALWI void negative_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_negative_, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC));
}
#else
ALWI void negative_tile_init() { MATH((llk::sfpu::UnaryDispatch<sfpu::Negative>::init().run())); }

// clang-format off
/**
 * Performs element-wise computation of the negative on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * The SHAPE template argument selects the region of the tile to negate (a
 * TensorShape with static storage); it defaults to the whole 32x32 tile.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <const TensorShape& SHAPE = DEFAULT_TENSOR_SHAPE>
ALWI void negative_tile(uint32_t idst) {
    MATH((llk::sfpu::UnaryDispatch<sfpu::Negative>::execute<DST_SYNC_MODE, SHAPE>(idst, idst).run()));
}

template <const TensorShape& SHAPE = DEFAULT_TENSOR_SHAPE>
ALWI void negative_tile_int32(uint32_t idst) {
    MATH((llk::sfpu::UnaryDispatch<sfpu::NegativeInt>::execute<DST_SYNC_MODE, SHAPE>(idst, idst).run()));
}
#endif

}  // namespace ckernel
