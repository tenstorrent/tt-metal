// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logaddexp2.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise log-sum-exp on two inputs: y = log2(2**x0 + 2**x1),
 * evaluated as max(x0, x1) + log2(1 + 2**-|x0 - x1|) so that no intermediate leaves
 * the representable range. Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst           | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void logaddexp2_binary_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_logaddexp2,
        (APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void logaddexp2_binary_tile_init() {
    // No SfpuType::logaddexp2 exists; `unused` selects no per-op init state.
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::calculate_sfpu_logaddexp2_init, (is_fp32_dest_acc_en))));
}

}  // namespace ckernel
