// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

// The kernel exists on Blackhole only; elsewhere the API is declared deleted, so a call fails to build.
#if defined(ARCH_BLACKHOLE)
#ifdef TRISC_MATH
#include "ckernel_sfpu_quick_gelu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise quick_gelu(x) = x/(1+exp(-1.702*x)) on each element of a tile in DST register at index
 * tile_index.
 *
 * BF16 (Float16_b) inputs, Blackhole only; accuracy is described in ckernel_sfpu_quick_gelu.h.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
#if defined(ARCH_BLACKHOLE)
ALWI void quick_gelu_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_quick_gelu, (APPROX, DST_ACCUM_MODE, 8), idst, VectorMode::RC));
}
#else
ALWI void quick_gelu_tile(uint32_t idst) = delete;
#endif

/**
 * Please refer to documentation for any_init.
 */
#if defined(ARCH_BLACKHOLE)
ALWI void quick_gelu_tile_init() {
    MATH((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::quick_gelu>(sfpu::quick_gelu_init)));
}
#else
ALWI void quick_gelu_tile_init() = delete;
#endif

}  // namespace ckernel
