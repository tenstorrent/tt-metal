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
 * Perform add + rsqrt operation: result = rsqrt(x + addend)
 *
 * @param idst The index of the tile in DST register buffer
 * @param addend The bit representation of a float to add before computing rsqrt
 */
template <bool fast_and_approx = false, VectorMode vec_mode = VectorMode::RC, int ITERATIONS = 8>
ALWI void add_rsqrt_tile(uint32_t idst, uint32_t addend) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_add_rsqrt,
        (APPROX, ITERATIONS, DST_ACCUM_MODE, fast_and_approx),
        idst,
        vec_mode,
        addend));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
