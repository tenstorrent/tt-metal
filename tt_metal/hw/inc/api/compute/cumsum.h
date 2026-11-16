// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_cumsum.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

enum class CumsumMode { Default, Compensated };

// clang-format off
// Cumulative sum
/**
 * Calculates the columnwise (top to bottom) cumulative sum.
 * For multi tile cumulative sum, tiles must come in NWH order (for example using reader_unary_transpose_wh_8bank) and
 * *first* must be false for all tiles where H != 0.
 * Tiles are also output in NWH order so writer_unary_transpose_wh can be used to store them back in NHW order.
 * The running column totals are carried between the tiles of one column in SFPU registers, so no other SFPU
 * operation may run between the *cumsum_tile* calls of the same column. After any other SFPU operation,
 * call *cumsum_tile_init* again before the next *cumsum_tile*, with the same mode.
 * CumsumMode::Compensated runs a compensated (Kahan) sum: the rounding error of each add is kept next to the
 * running totals and fed back into the next add, so the error does not grow with the length of the scan.
 * CumsumMode::Compensated is only available on Wormhole and Blackhole. Not supported on Quasar (compile error).
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | first           | Set true for tiles in the first row                                        | bool     |                                                       | False    |
 */
// clang-format on
template <CumsumMode mode = CumsumMode::Default>
ALWI void cumsum_tile(uint32_t idst, bool first = true) {
    // There is only non APPROXIMATE implementation; cumsum can only work in RC_custom mode.
    if constexpr (mode == CumsumMode::Compensated) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_cumsum_compensated,
            (false /* APPROXIMATE */, DST_ACCUM_MODE),
            idst,
            VectorMode::RC_custom,
            first));
    } else {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_cumsum,
            (false /* APPROXIMATE */),
            idst,
            VectorMode::RC_custom,
            first));
    }
}

/**
 * Please refer to documentation for any_init.
 */
template <CumsumMode mode = CumsumMode::Default>
ALWI void cumsum_tile_init() {
    // There is only non APPROXIMATE implementation
    if constexpr (mode == CumsumMode::Compensated) {
        MATH(SFPU_UNARY_INIT_FN(cumsum, sfpu::cumsum_compensated_init, (false /* APPROXIMATE */)));
    } else {
        MATH(SFPU_UNARY_INIT_FN(cumsum, sfpu::cumsum_init, (false /* APPROXIMATE */)));
    }
}

}  // namespace ckernel
