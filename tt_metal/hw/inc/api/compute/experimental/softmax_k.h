// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-row softmax over the first k columns (1 <= k <= 16) of a DST tile loaded via copy_tile.
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
 * @brief Initialize SFPU state for `softmax_k_tile` (pure SFPI exp needs no `exp_init`).
 */
ALWI void softmax_k_tile_init() { MATH(SFPU_UNARY_INIT(unused)); }

/**
 * @brief In-place per-row softmax over columns `[0, k)` of DST tile `idst`; `[k, 16)` are zeroed.
 * @tparam num_rows Active rows in face 0 (`1`, `2`, `4`, or `8`).
 * @tparam is_fp32_dest_acc_en FP32 DEST accumulation mode (defaults to `DST_ACCUM_MODE`).
 * @param idst DST tile index.
 * @param k Active column count (`1 <= k <= 16`).
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
