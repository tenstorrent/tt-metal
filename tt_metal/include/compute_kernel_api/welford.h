// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_welfords_sfpu_entry.h"
#endif
#include "debug/assert.h"

namespace ckernel {
/**
 * @brief Initializes the Welford's algorithm.
 * Programs the address mod and replay buffers for the Welford's algorithm.
 * Clears the previous mean and m2 values stored in the registers.
 * This call is blocking and is only available on the compute engine.
 */
ALWI void welford_init() {
    MATH((llk_math_welfords_sfpu_init()));
    MATH((llk_math_welfords_sfpu_clear_previous_mean_and_m2()));
}

/**
 * @brief Clears stale mean and m2 values stored in the registers.
 * This call is blocking and is only available on the compute engine.
 * This function should be called before calling `welford_update` for a new set of values.
 */
ALWI void welford_clear() { MATH((llk_math_welfords_sfpu_clear_previous_mean_and_m2())); }

/**
 * @brief Performs a Welford's online algorithm update for mean and m2 on a tile in the DST register
 *
 * This operation computes the running mean and m2 for a stream of data, enabling numerically stable
 * calculation of statistics in a single pass. The DST register buffer must be in acquired state via
 * @ref tile_regs_acquire call. This call is blocking and is only available on the compute engine.
 *
 * @tparam reciprocal_size The size of the reciprocal lookup table. If 0, the reciprocal will
 *                         be computed using float division instead.
 * @param input_dst_idx    The index of the tile in DST register buffer containing the new input.
 *                         Must be less than the size of the DST register. *
 * @param start_idx        The index of the first element in the tile; used to index the reciprocal
 *                         lookup table.
 * @param reciprocal_lut   The reference to the reciprocal lookup table. If an empty array is passed
 *                         the reciprocal will be computed using float division.
 *
 * @note All 32x32 elements (TILE_WIDTH * TILE_HEIGHT = 1024) of the input tile are processed by
 * this function.
 *
 * @return None. The mean and m2 values are held in the registers.
 */

template <uint32_t reciprocal_size>
ALWI void welford_update(
    uint32_t input_dst_idx, uint32_t start_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    // Check limits on the reciprocal lookup table.
    ASSERT((reciprocal_size == 0) || (start_idx < reciprocal_size));

    MATH((llk_math_welfords_sfpu_calculate_welfords_tile_<reciprocal_size>(input_dst_idx, start_idx, reciprocal_lut)));
}

/* -------------------------------------------------------------------------------------------------
 *  The below function is a flavor of *welford_update* that processes a subset of rows in the tile.
 *  Refer to the docstring of *welford_update* for more details.
 *  @param start_row The offset of the row to start from. Only rows starting from this offset are
 *                    processed in the tile. Should be 0 <= start_row <= 31.
 *  @param num_rows The number of rows to process. Should be 0 <= num_rows <= 32. Also,
 *                  0 <= start_row + num_rows <= 32.
 * -------------------------------------------------------------------------------------------------
 */
template <uint32_t reciprocal_size>
ALWI void welford_update_rows(
    uint32_t input_dst_idx,
    uint32_t start_idx,
    uint32_t start_row,
    uint32_t num_rows,
    const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    // Check limits on the reciprocal lookup table.
    ASSERT((reciprocal_size == 0) || (start_idx < reciprocal_size));

    // Check limits on the start row and number of rows.
    ASSERT(start_row + num_rows <= TILE_WIDTH);

    MATH((llk_math_welfords_sfpu_calculate_welfords_partial_tile_<reciprocal_size>(
        input_dst_idx, start_idx, start_row, num_rows, reciprocal_lut)));
}

/**
 * @brief Stores the mean and m2 values to the tile in the dst reg.
 *
 * This function stores the mean and m2 values to the tile in the dst reg. It is to be called to
 * temporarily store the mean and m2 values when using the SFPU for other calculations.
 * This call should be followed by a call to `welford_restore_state` to load the values back
 * into the SFPU when choosing to continue with the Welford's algorithm with the next set of values.
 * @param mean_dst_idx The index of the tile in the dst reg to store the mean values. The m2
 * values are stored in the consecutive tile after the mean.
 * @return None. The mean and m2 values are stored in the tile in the dst reg.
 */
ALWI void welford_save_state(uint32_t mean_dst_idx) {
    MATH((llk_math_welfords_sfpu_store_mean_m2_to_dst(mean_dst_idx)));
}

/**
 * @brief Loads the mean and m2 values from the tile in the dst reg into the SFPU.
 *
 * This function loads the mean and m2 values from the tile in the dst reg into the SFPU. It is to
 * be called after a call to `welford_save_state` to load the values back into the SFPU.
 * @param mean_dst_idx The index of the tile in the dst reg to load the mean values. The m2
 * values are loaded from the consecutive tile after the mean.
 * @return None. The mean and m2 values are loaded into the SFPU.
 */
ALWI void welford_restore_state(uint32_t mean_dst_idx) {
    MATH((llk_math_welfords_sfpu_load_mean_m2_from_dst(mean_dst_idx)));
}
/**
 * @brief Converts the accumulated M2 (sum of squares of differences from the mean) to variance and
 * stores the final mean and variance in the first row of the tiles in the dst reg.
 *
 * This function should be called after all elements of the input tile have been processed by
 * `welford_update`. It can also be called after a call to `welford_restore_state` to load
 * the mean and m2 values back into the SFPU. The DST register buffer must be in the acquired state
 * via @ref tile_regs_acquire.
 * This call is blocking and is only available on the compute engine.
 * @tparam reciprocal_size   The size of the reciprocal lookup table. If 0, the reciprocal will
 *                           be computed using float division.
 *
 * @param mean_dst_idx     The index of the tile in DST register buffer where the mean values will
 *                         be stored. The variance values are stored in the consecutive tile after
 *                         the mean. Must be less than the size of the DST register.
 * @param scale_idx        The index of the scale value to use for the variance calculation. This
 *                         value is used to convert the M2 to variance.
 * @param reciprocal_lut   The reference to the reciprocal lookup table. If an empty array is
 *                         passed (reciprocal_size is 0), the reciprocal will be computed using
 *                         float division.
 * @return                 None. The mean and variance tiles are updated in place. The first
 *                         row of each tile will hold the respective values.
 */
template <std::size_t reciprocal_size>
ALWI void welford_finalize_to_row(
    uint32_t mean_dst_idx, uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    // Check limits on the reciprocal lookup table.
    ASSERT((reciprocal_size == 0) || (scale_idx < reciprocal_size));

    MATH((llk_math_welfords_sfpu_store_mean_var_to_dst_row<reciprocal_size>(mean_dst_idx, scale_idx, reciprocal_lut)));
}

/**
 * @brief Stores the mean and variance values to the tile in the dst reg in the "raw" format.
 *
 * This function stores the mean and variance values to the tile in the dst reg in the "raw" format.
 * The function should be called to temporarily store the mean and variance values when using the
 * SFPU for other calculations. The DST register buffer must be in the acquired state via
 * @ref tile_regs_acquire. This call is blocking and is only available on the compute engine.
 * In raw format, the mean and variance values are stored in the first four rows of the first face
 * of the tile, with a stride of 2. Use `welford_finalize_to_row` if you need to store the
 * values in the first row of the tile.
 * @tparam reciprocal_size The size of the reciprocal lookup table. If 0, the reciprocal will
 *                         be computed using float division.
 * @param mean_dst_idx The index of the tile in the dst reg to store the mean values.
 *                     The variance values are stored in the consecutive tile after the mean.
 *                     Must be less than the size of the DST register.
 * @param scale_idx    The index of the scale value to use for the variance calculation. This
 *                     value is used to convert the M2 to variance.
 * @param reciprocal_lut The lookup table containing the reciprocals of the sample counts.
 * @return None. The mean and variance values are stored in the tile in the dst reg in the "raw"
 *         format. The first four rows of the first face of the tile will hold the values, with a
 *         stride of 2.
 */
template <std::size_t reciprocal_size>
ALWI void welford_finalize_to_face(
    uint32_t mean_dst_idx, uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    // Check limits on the reciprocal lookup table.
    ASSERT((reciprocal_size == 0) || (scale_idx < reciprocal_size));

    MATH((llk_math_welfords_sfpu_store_mean_var_to_dst_raw<reciprocal_size>(mean_dst_idx, scale_idx, reciprocal_lut)));
}

/* -------------------------------------------------------------------------------------------------
 * The below functions are flavors of above 3 to use with group_id argument
 * Refer to the docstring of the above 3 functions for more details.
 * @param group_id The group id to store the data for.
 * -------------------------------------------------------------------------------------------------
 */
ALWI void welford_save_state(uint32_t mean_dst_idx, uint32_t group_id) {
    MATH((llk_math_welfords_sfpu_store_mean_m2_to_dst(mean_dst_idx, group_id)));
}

ALWI void welford_restore_state(uint32_t mean_dst_idx, uint32_t group_id) {
    MATH((llk_math_welfords_sfpu_load_mean_m2_from_dst(mean_dst_idx, group_id)));
}

template <std::size_t reciprocal_size>
ALWI void welford_finalize_to_face(
    uint32_t mean_dst_idx,
    uint32_t group_id,
    uint32_t scale_idx,
    const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    // Check limits on the reciprocal lookup table.
    ASSERT((reciprocal_size == 0) || (scale_idx < reciprocal_size));

    MATH((llk_math_welfords_sfpu_store_mean_var_to_dst_raw<reciprocal_size>(
        mean_dst_idx, group_id, scale_idx, reciprocal_lut)));
}
}  // namespace ckernel
