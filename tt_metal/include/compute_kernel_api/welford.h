// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_welfords_sfpu_entry.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
/**
 * @brief Performs a Welford's online algorithm update for mean and m2 on a tile in the DST register.
 *
 * This operation computes the running mean and m2 for a stream of data, enabling numerically stable
 * calculation of statistics in a single pass. The DST register buffer must be in acquired state via @ref
 * tile_regs_acquire call. This call is blocking and is only available on the compute engine.
 *
 * @tparam input_dst_index   The index of the tile in DST register buffer containing the new input.
 *                           Must be less than the size of the DST register.
 * @tparam mean_dst_index    The index of the tile in DST register buffer containing the current running mean.
 *                           Must be less than the size of the DST register.
 *                           This tile can be left uninitialized when current_row is 0.
 * @tparam m2_dst_index      M2 is the short hand name for the sum of squares
 *                           The index of the tile in DST register buffer containing the running m2.
 *                           Must be less than the size of the DST register.
 *                           This tile can be left uninitialized when current_row is 0.
 * @tparam reformat_dst_to_col_on_end      When true, the dst reg at tile offset 1 and dst reg at tile offset 2 are
 * converted to column vectors when current_row + (samples processed in this call) == final_row. In addition, it also
 * converts M2 at tile offset 2 to variance.
 * @tparam reciprocal_size   The size of the reciprocal lookup table. If 0, the reciprocal will
 *                           be computed using float division instead.
 *
 * @param current_row     The current row index (starting from 0). Should follow 0 <= current_row <= TILE_HEIGHT (32)
 * and current_row
 *                        <= final_row. When current_row is 0, the previous mean and m2 are ignored.
 * @param final_row       The final row index. This index is not included in the update. Should follow current_row <=
 * final_row.A2D This dictates the total number of rows to update, starting from current_row.
 * @param num_skip_rows   Number of initial rows to skip in the update.
 *                        Setting this to a value greater than 0 skips the first num_skip_rows rows of the update.
 *                        Should follow 0 <= num_skip_rows <= TILE_HEIGHT (32).
 * @param reciprocal_lut  The reference to the reciprocal lookup table. If an empty array is passed (reciprocal_size is
 * 0), the reciprocal will be computed using float division.
 *
 * @note All TILE_WIDTH (32) columns of the input tile are processed by this function.
 *
 * @return None. Mean and m2 tiles are updated in place.
 */

template <
    uint32_t input_dst_index,
    uint32_t mean_dst_index,
    uint32_t m2_dst_index,
    bool reformat_dst_to_col_on_end,
    uint32_t reciprocal_size>
ALWI void welford_tile(
    uint32_t current_row,
    uint32_t final_row,
    uint32_t num_skip_rows,
    const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    MATH((llk_math_welfords_sfpu<
          input_dst_index,
          mean_dst_index,
          m2_dst_index,
          reformat_dst_to_col_on_end,
          /*convert_M2_to_var_on_end=*/false,
          reciprocal_size>(current_row, final_row, num_skip_rows, reciprocal_lut)));
}

/**
 * @brief Converts the accumulated M2 (sum of squares of differences from the mean) to variance and packs the final mean
 * and variance tiles into the DST register.
 *
 * This function should be called after all rows of the input tile have been processed by Welford's algorithm.
 * The DST register buffer must be in the acquired state via @ref tile_regs_acquire.
 * This call is blocking and is only available on the compute engine.
 * The function multiplies the M2 value in the DST register at tile offset 2 by the reciprocal of the scale factor to
 * compute the variance, and stores the result in the same location. Both the mean and variance are packed into the DST
 * register. TILE_WIDTH number of values are written to the first face of the DST register, with each value strided
 * by 2.
 *
 * @tparam mean_dst_index    The index of the tile in DST register buffer containing the means.
 *                           Must be less than the size of the DST register.
 * @tparam m2_dst_index      M2 is the short hand name for the sum of squares.
 *                           The index of the tile in DST register buffer containing the m2s.
 *                           Must be less than the size of the DST register.
 * @tparam reciprocal_size   The size of the reciprocal lookup table. If 0, the reciprocal will
 *                           be computed using float division.
 *
 * @param scale_factor       The reciprocal of this value (1/scale_factor) is multiplied with the M2 value in the DST
 * register at tile offset 2 to compute the variance.
 * @param reciprocal_lut     The reference to the reciprocal lookup table. If an empty array is passed (reciprocal_size
 * is 0), the reciprocal will be computed using float division.
 *
 * @return                   None. The mean and variance tiles are updated in place. TILE_WIDTH (32) number of values
 *                           are written to the DST register.
 *                           All TILE_WIDTH (32) number of values are written to the first face of the DST register.
 *                           Each valid value is followed by an invalid value.
 */
template <uint32_t mean_dst_index, uint32_t m2_dst_index, uint32_t reciprocal_size>
ALWI void welford_M2_to_var(uint32_t scale_factor, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    MATH((llk_math_welfords_sfpu<
          /*input_dst_index=*/0,
          mean_dst_index,
          m2_dst_index,
          /*reformat_dst_to_col_on_end=*/false,
          /*convert_M2_to_var_on_end=*/true,
          reciprocal_size>(scale_factor, /*final_row=*/0, /*num_skip_rows=*/0, reciprocal_lut)));
}

/**
 * Uses a copy of the ternery_sfpu_init
 * Programs the replay for fast LLK. Needed otherwise LLK wont work
 */
ALWI void welford_init() { MATH((llk_math_welfords_sfpu_init())); }

}  // namespace ckernel
