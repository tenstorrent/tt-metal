// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
// Blackhole-only: the add_rsqrt SFPU functor lives only in the Blackhole llk_api tree.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

/**
 * Initialize for add + rsqrt operation: result = rsqrt(x + addend)
 * Useful for operations like RMSNorm: rsqrt(variance + epsilon)
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_rsqrt_tile_init() {
    // The init only programs the rsqrt constants; it does not depend on fast_and_approx / ITERATIONS.
    MATH((sfpu::AddRsqrt<APPROX, false /*FAST_APPROX*/, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

/**
 * Perform add + rsqrt operation: result = rsqrt(x + addend)
 *
 * @param idst The index of the tile in DST register buffer
 * @param addend The bit representation of a float to add before computing rsqrt
 */
template <
    bool fast_and_approx = false,
    VectorMode vec_mode = VectorMode::RC,
    int ITERATIONS = 8,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_rsqrt_tile(uint32_t idst, uint32_t addend) {
    MATH((sfpu::AddRsqrt<APPROX, fast_and_approx, DST_SYNC_MODE, is_fp32_dest_acc_en, ITERATIONS>::calculate(
        idst, vec_mode, addend)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
