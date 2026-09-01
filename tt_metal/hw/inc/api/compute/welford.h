// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <cstdint>

#include "api/compute/common_globals.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_welfords_sfpu_entry.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif
#include "api/debug/assert.h"

namespace ckernel {
/**
 * Controls whether `welford_init` resets the running mean and M2 accumulators.
 *
 * ClearStats (default): Clears the previous mean and M2 values stored in the registers. Use for a
 *     fresh Welford pass.
 * PreserveStats: Leaves the running mean and M2 accumulators (LREG4/5) intact. Use when re-arming
 *     the SFPU replay buffer mid-pass after another op (e.g. `transpose_tile` on the
 *     unpack-to-DEST fp32 path) has clobbered the welford recurrence slots.
 */
enum class WelfordInitMode : std::uint8_t {
    ClearStats,
    PreserveStats,
};

/**
 * @brief Initializes the Welford's algorithm.
 * Programs the address mod and replay buffers for the Welford's algorithm.
 * Clears the previous mean and m2 values stored in the registers when `mode` is `ClearStats`.
 * This call is blocking and is only available on the compute engine.
 * @tparam mode Controls whether the running mean and M2 accumulators are cleared.
 *              See @ref WelfordInitMode.
 */
template <WelfordInitMode mode = WelfordInitMode::ClearStats>
ALWI void welford_init() {
    MATH((llk_math_welfords_sfpu_init()));
    if constexpr (mode == WelfordInitMode::ClearStats) {
        MATH((llk_math_welfords_sfpu_clear_previous_mean_and_m2()));
    }
}

/**
 * @brief Re-establish UNPACK and MATH state for Welford after another op has reconfigured them
 * (e.g. FPU `mul_tiles_bcast_scalar`). Does not reprogram the SFPU replay buffer or clear the
 * running mean/M2 accumulators in LREG4/5. Example usage of this is in `welford_update` - this is called once per tile
 * when the `do_scale` path runs `mul_tiles_bcast_scalar` in the same DST window.
 */
#ifndef ARCH_QUASAR
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void welford_reinit(std::uint32_t cbid, std::uint32_t call_line = __builtin_LINE()) {
    state_configure(cbid, call_line);
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        /*transpose=*/0, /*transpose_within_16x16_face=*/false, cbid)));
    MATH((llk_math_welfords_sfpu_reinit<is_fp32_dest_acc_en>(cbid)));
}
#endif

/**
 * @brief Clears stale mean and m2 values stored in the registers.
 * This call is blocking and is only available on the compute engine.
 * This function should be called before calling `welford_update` for a new set of values.
 */
ALWI void welford_clear() { MATH((llk_math_welfords_sfpu_clear_previous_mean_and_m2())); }

/**
 * @brief Initializes the SFPU state used by the two-pass statistics helpers.
 *
 * Configures the Welford address mode used by their SFPU loads without
 * programming Welford's unused replay buffer. The first shifted update with
 * `initialize_anchor=true` initializes the accumulators.
 */
ALWI void two_pass_stats_init_shifted() { MATH((llk_math_two_pass_sfpu_init())); }

/**
 * @brief Accumulates a row range into the current two-pass sum or centred-M2 state.
 * @tparam accumulate_m2 If true, accumulates squared differences from the current mean; otherwise accumulates sums.
 * @tparam dual_m2 If true, uses two independent accumulators to hide SFPU dependency latency.
 * @param input_dst_idx Index of the input tile in the DST register buffer.
 * @param start_row First tile row to process.
 * @param num_rows Number of consecutive tile rows to process.
 */
template <bool accumulate_m2, bool dual_m2 = true>
ALWI void two_pass_stats_update_rows(std::uint32_t input_dst_idx, std::uint32_t start_row, std::uint32_t num_rows) {
    ASSERT(start_row + num_rows <= TILE_WIDTH);
    MATH((llk_math_two_pass_sfpu_update_rows<accumulate_m2, dual_m2>(input_dst_idx, start_row, num_rows)));
}

/**
 * @brief Accumulates shifted sums or centred M2 over a row range.
 * @tparam accumulate_m2 If true, accumulates centred squared residuals; otherwise accumulates shifted sums.
 * @tparam initialize_anchor If true, initialises the common anchor from the first selected input value.
 * @tparam dual_accumulator If true, uses two independent accumulators to hide SFPU dependency latency.
 * @param input_dst_idx Index of the input tile in the DST register buffer.
 * @param start_row First tile row to process.
 * @param num_rows Number of consecutive tile rows to process.
 */
template <bool accumulate_m2, bool initialize_anchor = false, bool dual_accumulator = true>
ALWI void two_pass_stats_update_shifted_rows(
    std::uint32_t input_dst_idx, std::uint32_t start_row, std::uint32_t num_rows) {
    ASSERT(start_row + num_rows <= TILE_WIDTH);
    MATH((llk_math_two_pass_sfpu_update_shifted_rows<accumulate_m2, initialize_anchor, dual_accumulator>(
        input_dst_idx, start_row, num_rows)));
}

/**
 * @brief Converts the shifted sum into a mean and clears the accumulators for the centred-M2 pass.
 * @tparam dual_sum If true, combines the two shifted-sum accumulators before scaling.
 * @tparam retain_anchor If true, retains the anchor for the subsequent centred-M2 pass.
 * @param reciprocal_bits Bit representation of the FP32 reciprocal population count.
 */
template <bool dual_sum = true, bool retain_anchor = false>
ALWI void two_pass_stats_finish_shifted_mean(std::uint32_t reciprocal_bits) {
    MATH((llk_math_two_pass_sfpu_finish_shifted_mean<dual_sum, retain_anchor>(reciprocal_bits)));
}

/** @brief Clears the two-pass mean, sum, and M2 accumulator registers. */
ALWI void two_pass_stats_clear() { MATH((llk_math_two_pass_sfpu_clear_stats())); }

/**
 * @brief Stores the current mean and M2 state in consecutive DST tiles.
 * @tparam dual_m2 If true, combines the two M2 accumulators before storing.
 * @param mean_dst_idx Index of the DST tile that receives the mean; M2 is stored in the following tile.
 */
template <bool dual_m2 = true>
ALWI void two_pass_stats_save_state(std::uint32_t mean_dst_idx) {
    MATH((llk_math_two_pass_sfpu_store_mean_m2_to_dst<dual_m2>(mean_dst_idx)));
}

/**
 * @brief Finalises variance while storing the anchor and mean correction in a split-mean row layout.
 * @tparam dual_m2 If true, combines the two M2 accumulators before scaling.
 * @param mean_dst_idx Index of the DST tile that receives the split mean; variance uses the following tile.
 * @param reciprocal_bits Bit representation of the FP32 reciprocal population count.
 */
template <bool dual_m2 = true>
ALWI void two_pass_stats_finalize_split_mean_to_row(std::uint32_t mean_dst_idx, std::uint32_t reciprocal_bits) {
    MATH((llk_math_two_pass_sfpu_store_split_mean_var_to_dst_row<dual_m2>(mean_dst_idx, reciprocal_bits)));
}

/**
 * @brief Stores the current shift anchor in a DST tile.
 * @param anchor_dst_idx Index of the destination tile.
 */
ALWI void two_pass_stats_save_anchor(std::uint32_t anchor_dst_idx) {
    MATH((llk_math_two_pass_sfpu_store_anchor_to_dst(anchor_dst_idx)));
}

/**
 * @brief Restores the shift anchor from a DST tile.
 * @param anchor_dst_idx Index of the source tile.
 */
ALWI void two_pass_stats_restore_anchor(std::uint32_t anchor_dst_idx) {
    MATH((llk_math_two_pass_sfpu_load_anchor_from_dst(anchor_dst_idx)));
}

/**
 * @brief Stores the shift anchor in the anchor row of a saved statistics state.
 * @param mean_dst_idx Index of the saved mean/M2 state.
 */
ALWI void two_pass_stats_save_anchor_to_state(std::uint32_t mean_dst_idx) {
    MATH((llk_math_two_pass_sfpu_store_anchor_to_state_dst(mean_dst_idx)));
}

/**
 * @brief Restores the shift anchor from the anchor row of a saved statistics state.
 * @param mean_dst_idx Index of the saved mean/M2 state.
 */
ALWI void two_pass_stats_restore_anchor_from_state(std::uint32_t mean_dst_idx) {
    MATH((llk_math_two_pass_sfpu_load_anchor_from_state_dst(mean_dst_idx)));
}

/**
 * @brief Merges the current block's mean/M2 with a previously saved statistics state.
 * @tparam dual_m2 If true, combines the current block's two M2 accumulators before merging.
 * @param mean_dst_idx Index of the saved mean/M2 state to update.
 * @param total_reciprocal_bits Bit representation of the reciprocal combined population count.
 * @param block_n_bits Bit representation of the current block's FP32 population count.
 */
template <bool dual_m2 = true>
ALWI void two_pass_stats_combine_block(
    std::uint32_t mean_dst_idx, std::uint32_t total_reciprocal_bits, std::uint32_t block_n_bits) {
    MATH((llk_math_two_pass_sfpu_combine_block_to_dst<dual_m2>(mean_dst_idx, total_reciprocal_bits, block_n_bits)));
}

/**
 * @brief Finalises the current mean and variance into the first row of consecutive DST tiles.
 * @tparam dual_m2 If true, combines the two M2 accumulators before scaling.
 * @param mean_dst_idx Index of the mean tile; variance is stored in the following tile.
 * @param reciprocal_bits Bit representation of the FP32 reciprocal population count.
 */
template <bool dual_m2 = true>
ALWI void two_pass_stats_finalize_to_row(std::uint32_t mean_dst_idx, std::uint32_t reciprocal_bits) {
    MATH((llk_math_two_pass_sfpu_store_mean_var_to_dst_row<dual_m2>(mean_dst_idx, reciprocal_bits)));
}

/**
 * @brief Finalises one group's mean and variance into the raw face layout of consecutive DST tiles.
 * @tparam dual_m2 If true, combines the two M2 accumulators before scaling.
 * @param mean_dst_idx Index of the mean tile; variance is stored in the following tile.
 * @param group_id Group slot within the raw face layout.
 * @param reciprocal_bits Bit representation of the FP32 reciprocal population count.
 */
template <bool dual_m2 = true>
ALWI void two_pass_stats_finalize_to_face(
    std::uint32_t mean_dst_idx, std::uint32_t group_id, std::uint32_t reciprocal_bits) {
    MATH((llk_math_two_pass_sfpu_store_mean_var_to_dst_raw<dual_m2>(mean_dst_idx, group_id, reciprocal_bits)));
}

#ifdef WELFORD_SFPU_LOCAL_COMBINE
/**
 * @brief Finalises one group's local statistics and combines its lane populations into the raw face layout.
 * @tparam dual_m2 If true, combines the two M2 accumulators before finalisation.
 * @param mean_dst_idx Index of the mean tile; variance is stored in the following tile.
 * @param group_id Group slot within the raw face layout.
 * @param reciprocal_bits Bit representation of the FP32 reciprocal local population count.
 */
template <bool dual_m2 = true>
ALWI void two_pass_stats_finalize_and_combine_to_face(
    std::uint32_t mean_dst_idx, std::uint32_t group_id, std::uint32_t reciprocal_bits) {
    MATH((llk_math_two_pass_sfpu_store_combined_mean_var_to_dst_raw<dual_m2>(mean_dst_idx, group_id, reciprocal_bits)));
}
#endif

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

template <std::uint32_t reciprocal_size>
ALWI void welford_update(
    std::uint32_t input_dst_idx,
    std::uint32_t start_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
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
template <std::uint32_t reciprocal_size>
ALWI void welford_update_rows(
    std::uint32_t input_dst_idx,
    std::uint32_t start_idx,
    std::uint32_t start_row,
    std::uint32_t num_rows,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
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
ALWI void welford_save_state(std::uint32_t mean_dst_idx) {
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
ALWI void welford_restore_state(std::uint32_t mean_dst_idx) {
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
    std::uint32_t mean_dst_idx,
    std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
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
    std::uint32_t mean_dst_idx,
    std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
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
ALWI void welford_save_state(std::uint32_t mean_dst_idx, std::uint32_t group_id) {
    MATH((llk_math_welfords_sfpu_store_mean_m2_to_dst(mean_dst_idx, group_id)));
}

ALWI void welford_restore_state(std::uint32_t mean_dst_idx, std::uint32_t group_id) {
    MATH((llk_math_welfords_sfpu_load_mean_m2_from_dst(mean_dst_idx, group_id)));
}

/**
 * @brief Saves the active group state and restores another group state.
 * @param mean_dst_idx Index of the mean state tile; M2 is stored in the following tile.
 * @param save_group_id Group slot that receives the active state.
 * @param restore_group_id Group slot to load into the SFPU accumulators.
 */
ALWI void two_pass_stats_switch_group(
    std::uint32_t mean_dst_idx, std::uint32_t save_group_id, std::uint32_t restore_group_id) {
    MATH((llk_math_two_pass_sfpu_switch_group(mean_dst_idx, save_group_id, restore_group_id)));
}

template <std::size_t reciprocal_size>
ALWI void welford_finalize_to_face(
    std::uint32_t mean_dst_idx,
    std::uint32_t group_id,
    std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    // Check limits on the reciprocal lookup table.
    ASSERT((reciprocal_size == 0) || (scale_idx < reciprocal_size));

    MATH((llk_math_welfords_sfpu_store_mean_var_to_dst_raw<reciprocal_size>(
        mean_dst_idx, group_id, scale_idx, reciprocal_lut)));
}
}  // namespace ckernel
