// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "tensor_shape.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
#ifdef ARCH_QUASAR
ALWI void sqrt_tile_init() { MATH(SFPU_UNARY_INIT_FN(sqrt, sfpu::sqrt_init, (APPROX))); }

template <bool FAST_APPROX = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sqrt_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sqrt,
        (APPROX, 8 /*ITERATIONS*/, is_fp32_dest_acc_en, FAST_APPROX),
        idst,
        VectorMode::RC));
}
#else
/**
 * Please refer to documentation for any_init.
 */
ALWI void sqrt_tile_init() { MATH(llk::sfpu::UnaryDispatch<sfpu::Sqrt>::init().run<APPROX>()); }

// clang-format off
/**
 * Performs element-wise computation of the square root on each element of a
 * tile in DST register at index idst. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * The SHAPE template argument selects the region of the tile to process (a
 * TensorShape with static storage); it defaults to the whole 32x32 tile.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <
    bool FAST_APPROX = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    const TensorShape& SHAPE = DEFAULT_TENSOR_SHAPE>
ALWI void sqrt_tile(uint32_t idst) {
    MATH(llk::sfpu::UnaryDispatch<sfpu::Sqrt>::execute<DST_SYNC_MODE, SHAPE>(idst, idst)
             .template run<APPROX, is_fp32_dest_acc_en, FAST_APPROX>());
}
#endif

}  // namespace ckernel
