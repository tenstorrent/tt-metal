// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute API for per-row softmax over the first k columns (1 <= k <= 16) of an (M, 16) or
// (M, 32) tile (M in {1, 2, 4, 8}) loaded into DST via standard copy_tile.
#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_sfpu/ckernel_sfpu_softmax_k.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

/**
 * @brief Initialize SFPU address modes for `softmax_k_tile` on the math thread.
 * @note Call before `softmax_k_tile`. Uses standard `copy_tile` to load the input into DST.
 *       Pure SFPI exp leaves materialize constants inline via SFPLOADI, so `SfpuType::unused`
 *       init (config reg + ADDR_MOD_7 + counter reset) suffices without `exp_init`.
 */
ALWI void softmax_k_tile_init() { MATH(SFPU_UNARY_INIT(unused)); }

/**
 * @brief In-place per-row softmax over the first `k` columns of DST tile `idst`.
 *
 * Columns `[k, 16)` are masked out before max and written as `0.0`. Works for unsorted or
 * presorted inputs, zero logits inside `[0, k)`, and arbitrary `k` in `[1, 16]`.
 *
 * @tparam num_rows Number of independent rows packed in the tile (`1`, `2`, `4`, or `8`).
 * @tparam is_fp32_dest_acc_en Whether FP32 DEST accumulation is enabled (defaults to `DST_ACCUM_MODE`).
 * @param idst DST tile index in the acquired DEST section.
 * @param k Number of active columns per row (`1 <= k <= 16`).
 */
template <std::uint32_t num_rows, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void softmax_k_tile(std::uint32_t idst, std::uint32_t k) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_softmax_k_,
        (is_fp32_dest_acc_en, num_rows),
        idst,
        VectorMode::RC_custom,
        k));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
