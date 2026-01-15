// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_add_rsqrt.h"
#endif

namespace ckernel {

/**
 * Initialize for add + rsqrt operation: result = rsqrt(x + addend)
 * Useful for operations like RMSNorm: rsqrt(variance + epsilon)
 */
ALWI void add_rsqrt_tile_init() { MATH((llk_math_eltwise_unary_sfpu_add_rsqrt_init<APPROX>())); }

/**
 * Perform add + rsqrt operation: result = rsqrt(x + addend)
 *
 * @param idst The index of the tile in DST register buffer
 * @param addend The bit representation of a float to add before computing rsqrt
 */
template <bool fast_and_approx = false, int vec_mode = VectorMode::RC, int ITERATIONS = 8>
ALWI void add_rsqrt_tile(uint32_t idst, uint32_t addend) {
    MATH((llk_math_eltwise_unary_sfpu_add_rsqrt<APPROX, DST_ACCUM_MODE, fast_and_approx, ITERATIONS>(
        idst, addend, vec_mode)));
}

}  // namespace ckernel
