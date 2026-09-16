// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#ifdef TRISC_MATH
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa_fw.h"
#endif

// First-column SFPU variants for the column vectors produced by ReduceDim::REDUCE_ROW.
//
// A REDUCE_ROW output only carries meaningful data in column 0 of each row. Column 0 lives
// in faces 0 and 2; faces 1 and 3 hold columns 16-31 and are never needed. VectorMode::C
// visits only faces 0 and 2, and a stride-2 dst_reg walk covers the column-0 half of each
// face, so these ops cost 2 faces x 4 iterations = 8 SFPU iterations instead of the
// full-tile 4 x 8 = 32.
//
// Both variants pair with the standard full-tile inits (sqrt_tile_init, recip_tile_init);
// only the per-tile traversal changes.

#ifdef TRISC_MATH
namespace _first_column_detail {

// sqrt over ITERATIONS SFPU vectors, advancing dst_reg by DST_STRIDE each step.
//
// Calls the same _calculate_sqrt_body_ that _calculate_sqrt_internal_ uses, with the same
// template arguments sqrt_tile passes (calculate_sqrt defaults legacy_compat to false, so
// sqrt_tile takes the _internal_ path). The lanes this touches therefore get results
// identical to sqrt_tile.
template <int ITERATIONS, bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool FAST_APPROX, int DST_STRIDE = 1>
inline void sfpi_sqrt() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat tmp = ckernel::sfpu::_calculate_sqrt_body_<APPROXIMATION_MODE, /*RECIPROCAL*/ false, FAST_APPROX>(
            sfpi::dst_reg[0]);
        if constexpr (!is_fp32_dest_acc_en) {
            tmp = sfpi::convert<sfpi::vFloat16b>(tmp, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = tmp;
        if constexpr (DST_STRIDE == 1) {
            sfpi::dst_reg++;
        } else {
            sfpi::dst_reg += DST_STRIDE;
        }
    }
}

}  // namespace _first_column_detail
#endif  // TRISC_MATH

// First-column sqrt (2 faces x 4 half-face iterations, VectorMode::C).
// Pair with the standard sqrt_tile_init(), which programs the constants the body reads.
template <bool FAST_APPROX = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
inline void sqrt_tile_first_column(uint32_t idst) {
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_params_(
        _first_column_detail::sfpi_sqrt</*ITERATIONS*/ 4, APPROX, is_fp32_dest_acc_en, FAST_APPROX, /*DST_STRIDE*/ 2>,
        idst,
        VectorMode::C);
#endif  // TRISC_MATH
}

// First-column reciprocal (2 faces x 4 half-face iterations, VectorMode::C).
// Moved here from sdpa_fw's private kernel utils, unchanged apart from moving the
// TRISC_MATH guard inside the body so it matches sqrt_tile_first_column above and can be
// called without a MATH(( )) wrapper. Pair with the standard recip_tile_init().
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
inline void recip_tile_first_column(uint32_t idst) {
#ifdef TRISC_MATH
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_recip_first_column, (is_fp32_dest_acc_en), idst, VectorMode::C);
#endif  // TRISC_MATH
}
