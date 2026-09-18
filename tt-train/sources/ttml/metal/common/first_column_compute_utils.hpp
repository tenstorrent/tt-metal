// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/sqrt.h"

// First-column sqrt for the column vectors produced by ReduceDim::REDUCE_ROW.
//
// A REDUCE_ROW output only carries meaningful data in column 0 of each row. Column 0 lives
// in faces 0 and 2; faces 1 and 3 hold columns 16-31 and are never needed. VectorMode::C
// visits only faces 0 and 2, and a stride-2 dst_reg walk covers the column-0 half of each
// face, so this costs 2 faces x 4 iterations = 8 SFPU iterations instead of the full-tile
// 4 x 8 = 32.
//
// Pairs with the standard sqrt_tile_init(); only the per-tile traversal changes.

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// LLK-style body, shaped like calculate_recip_first_column in
// experimental/llk_sfpu/ckernel_sfpu_sdpa_fw.h: 4 half-face iterations at dst_reg stride 2.
//
// Calls the same _calculate_sqrt_body_ that _calculate_sqrt_internal_ uses, with the same
// template arguments sqrt_tile passes (calculate_sqrt defaults legacy_compat to false, so
// sqrt_tile takes the _internal_ path). The lanes this touches therefore get results
// identical to sqrt_tile.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool FAST_APPROX>
inline void calculate_sqrt_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat tmp =
            _calculate_sqrt_body_<APPROXIMATION_MODE, /*RECIPROCAL*/ false, FAST_APPROX>(sfpi::dst_reg[0]);
        if constexpr (!is_fp32_dest_acc_en) {
            tmp = sfpi::convert<sfpi::vFloat16b>(tmp, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = tmp;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH

namespace ckernel {

// First-column sqrt (2 faces x 4 half-face iterations, VectorMode::C). Same shape as
// sqrt_tile: SFPU_UNARY_CALL runs the dst_index bounds check before the LLK params wrapper.
template <bool FAST_APPROX = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sqrt_tile_first_column(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sqrt_first_column,
        (APPROX, is_fp32_dest_acc_en, FAST_APPROX),
        idst,
        VectorMode::C));
}

}  // namespace ckernel
