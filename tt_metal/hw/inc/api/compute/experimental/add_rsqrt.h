// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
// Blackhole-only: the add_rsqrt SFPU functor lives only in the Blackhole llk_api tree.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

/**
 * Initialize for add + rsqrt operation: result = rsqrt(x + addend)
 * Useful for operations like RMSNorm: rsqrt(variance + epsilon)
 */
ALWI void add_rsqrt_tile_init() { MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::init_add_rsqrt, (APPROX))); }

/**
 * Perform scale + add + rsqrt operation: result = rsqrt(x * input_scale + addend)
 *
 * typed_bf16_store=true selects the direct BF16 store used by fused normalization.
 * input_scale applies a factor such as RMSNorm's 1/N at fp32, for a caller that
 * reduces sum(x^2) unscaled: folded into the reduce it would pass through a bf16
 * SrcB fill.
 *
 * @param idst The index of the tile in DST register buffer
 * @param addend The bit representation of a float to add before computing rsqrt
 * @tparam input_scale The bit representation of a float that multiplies x first (1.0 by default)
 */
template <
    bool fast_and_approx = false,
    VectorMode vec_mode = VectorMode::RC,
    int ITERATIONS = 8,
    bool typed_bf16_store = false,
    uint32_t input_scale = 0x3f800000u>
ALWI void add_rsqrt_tile(uint32_t idst, uint32_t addend) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_add_rsqrt,
        (APPROX, ITERATIONS, DST_ACCUM_MODE, fast_and_approx, typed_bf16_store, input_scale),
        idst,
        vec_mode,
        addend));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
