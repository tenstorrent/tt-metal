// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_welfords.h"

namespace ckernel {

// Compute API entry points for welford.h, named as on Blackhole / Wormhole. Each runs the Quasar kernel
// on the Dest tile it names (SFPU_UNARY_CALL sets the Dest base, as _llk_math_welfords_sfpu_params_ does
// there). The running statistics live in LREG4-7 between calls, so no other SFPU op may run in between.

namespace welfords_detail {

// Saved or finalised statistics span the mean tile and the next one; the combined two-pass store also
// clobbers a third tile as scratch.
constexpr std::uint32_t STATS_TILES = 2;
constexpr std::uint32_t COMBINED_STATS_TILES = 3;

template <std::uint32_t NUM_TILES>
inline void check_consecutive_dst(const std::uint32_t first_dst_idx) {
    constexpr std::uint32_t max_dest_tiles =
        trisc::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, trisc::DstTileShape::Tile32x32>();
    LLK_ASSERT(first_dst_idx + NUM_TILES <= max_dest_tiles, "welford statistics: the Dest tiles must fit in Dest");
}

}  // namespace welfords_detail

// ---- Welford (one pass) ---------------------------------------------------------------------------

inline void llk_math_welfords_sfpu_init() { llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(sfpu::welfords_init); }

inline void llk_math_welfords_sfpu_clear_previous_mean_and_m2() { sfpu::welfords_clear_previous_mean_and_m2(); }

template <std::uint32_t reciprocal_size>
inline void llk_math_welfords_sfpu_calculate_welfords_tile_(
    const std::uint32_t input_dst_idx,
    const std::uint32_t start_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_welfords,
        (false /*PARTIAL_TILE*/, reciprocal_size),
        input_dst_idx,
        VectorMode::RC_custom,
        start_idx,
        reciprocal_lut,
        0u /*start_row*/,
        static_cast<std::uint32_t>(TILE_R_DIM) /*num_rows*/);
}

template <std::uint32_t reciprocal_size>
inline void llk_math_welfords_sfpu_calculate_welfords_partial_tile_(
    const std::uint32_t input_dst_idx,
    const std::uint32_t start_idx,
    const std::uint32_t start_row,
    const std::uint32_t num_rows,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_welfords,
        (true /*PARTIAL_TILE*/, reciprocal_size),
        input_dst_idx,
        VectorMode::RC_custom,
        start_idx,
        reciprocal_lut,
        start_row,
        num_rows);
}

/** @brief Save the running mean to this tile and M2 to the next; the group_id overload uses that group slot. */
inline void llk_math_welfords_sfpu_store_mean_m2_to_dst(const std::uint32_t mean_dst_idx) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_store_mean_m2_to_dst,
        (false /*GROUPED*/),
        mean_dst_idx,
        VectorMode::RC_custom,
        0u /*group_id*/);
}

inline void llk_math_welfords_sfpu_store_mean_m2_to_dst(
    const std::uint32_t mean_dst_idx, const std::uint32_t group_id) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_store_mean_m2_to_dst,
        (true /*GROUPED*/),
        mean_dst_idx,
        VectorMode::RC_custom,
        group_id);
}

/** @brief Restore the state saved by @ref llk_math_welfords_sfpu_store_mean_m2_to_dst. */
inline void llk_math_welfords_sfpu_load_mean_m2_from_dst(const std::uint32_t mean_dst_idx) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_load_mean_m2_from_dst,
        (false /*GROUPED*/),
        mean_dst_idx,
        VectorMode::RC_custom,
        0u /*group_id*/);
}

inline void llk_math_welfords_sfpu_load_mean_m2_from_dst(
    const std::uint32_t mean_dst_idx, const std::uint32_t group_id) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_load_mean_m2_from_dst,
        (true /*GROUPED*/),
        mean_dst_idx,
        VectorMode::RC_custom,
        group_id);
}

/** @brief Write the mean as row 0 of this tile and the variance as row 0 of the next; ends the running state. */
template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_row(
    const std::uint32_t mean_dst_idx,
    const std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_store_mean_var_to_dst,
        (sfpu::WelfordsOutputLayout::Row, false /*GROUPED*/, reciprocal_size),
        mean_dst_idx,
        VectorMode::RC_custom,
        scale_idx,
        reciprocal_lut,
        0u /*group_id*/);
}

/** @brief Write the mean to this tile and the variance to the next, raw face layout; ends the running state. */
template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_raw(
    const std::uint32_t mean_dst_idx,
    const std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_store_mean_var_to_dst,
        (sfpu::WelfordsOutputLayout::Face, false /*GROUPED*/, reciprocal_size),
        mean_dst_idx,
        VectorMode::RC_custom,
        scale_idx,
        reciprocal_lut,
        0u /*group_id*/);
}

template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_raw(
    const std::uint32_t mean_dst_idx,
    const std::uint32_t group_id,
    const std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_store_mean_var_to_dst,
        (sfpu::WelfordsOutputLayout::Face, true /*GROUPED*/, reciprocal_size),
        mean_dst_idx,
        VectorMode::RC_custom,
        scale_idx,
        reciprocal_lut,
        group_id);
}

// ---- Shifted two-pass statistics ------------------------------------------------------------------

/**
 * @brief Configure the SFPU for two-pass statistics.
 *
 * @note Call @ref llk_math_two_pass_sfpu_clear_stats before accumulating a new population; initialisation does not
 * clear the running register state.
 */
inline void llk_math_two_pass_sfpu_init() { llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(); }

/** @brief Clear the active mean, shifted-sum, and M2 register state. */
inline void llk_math_two_pass_sfpu_clear_stats() { sfpu::_two_pass_clear_stats_(); }

/**
 * @brief Accumulate centred squared differences from a contiguous row range of one DST tile.
 *
 * @tparam dual_m2: Use independent accumulators for alternating rows.
 * @param input_dst_idx: DST tile containing the transposed input.
 * @param start_row: First row to include.
 * @param num_rows: Number of rows to include.
 */
template <bool dual_m2>
inline void llk_math_two_pass_sfpu_update_rows(
    const std::uint32_t input_dst_idx, const std::uint32_t start_row, const std::uint32_t num_rows) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _two_pass_update_rows_,
        (dual_m2),
        input_dst_idx,
        VectorMode::RC_custom,
        start_row,
        num_rows);
}

/**
 * @brief Accumulate either shifted pass-one sums or centred pass-two M2 over one DST tile.
 *
 * @tparam accumulate_m2: Select the M2 pass instead of the shifted-sum pass.
 * @tparam initialize_anchor: Initialise the common anchor from the first selected input.
 * @tparam dual_accumulator: Use independent accumulators for alternating rows.
 * @param input_dst_idx: DST tile containing the transposed input.
 * @param start_row: First row to include.
 * @param num_rows: Number of rows to include.
 */
template <bool accumulate_m2, bool initialize_anchor, bool dual_accumulator>
inline void llk_math_two_pass_sfpu_update_shifted_rows(
    const std::uint32_t input_dst_idx, const std::uint32_t start_row, const std::uint32_t num_rows) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _two_pass_update_shifted_rows_,
        (accumulate_m2, initialize_anchor, dual_accumulator),
        input_dst_idx,
        VectorMode::RC_custom,
        start_row,
        num_rows);
}

/**
 * @brief Convert the shifted sums to a mean and reset the M2 accumulators for pass two.
 *
 * @tparam dual_sum: Fold the secondary sum accumulator before finalising.
 * @tparam retain_anchor: Preserve the original anchor for compensated normalisation.
 * @param reciprocal_bits: FP32 bit pattern for the reciprocal population count.
 */
template <bool dual_sum, bool retain_anchor = false>
inline void llk_math_two_pass_sfpu_finish_shifted_mean(const std::uint32_t reciprocal_bits) {
    sfpu::_two_pass_finish_shifted_mean_<dual_sum, retain_anchor>(reciprocal_bits);
}

/**
 * @brief Spill the current mean and M2 into two consecutive DST tiles.
 *
 * @tparam dual_m2: Fold the secondary M2 accumulator before storing.
 * @param mean_dst_idx: First of the consecutive statistics DST tiles.
 */
template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_mean_m2_to_dst(const std::uint32_t mean_dst_idx) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _two_pass_store_mean_m2_to_dst_, (dual_m2), mean_dst_idx, VectorMode::RC_custom);
}

/**
 * @brief Store the retained anchor, anchor-minus-mean correction, and variance for compensated normalisation.
 *
 * @tparam dual_m2: Fold the secondary M2 accumulator before storing.
 * @param mean_dst_idx: First of the consecutive statistics DST tiles.
 * @param reciprocal_bits: FP32 bit pattern for the reciprocal population count.
 */
template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_split_mean_var_to_dst_row(
    const std::uint32_t mean_dst_idx, const std::uint32_t reciprocal_bits) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _two_pass_store_split_mean_var_to_dst_row_,
        (dual_m2),
        mean_dst_idx,
        VectorMode::RC_custom,
        reciprocal_bits);
}

/**
 * @brief Store the retained anchor at raw offset zero of the selected DST tile.
 *
 * @param anchor_dst_idx: Destination tile for the anchor.
 */
inline void llk_math_two_pass_sfpu_store_anchor_to_dst(const std::uint32_t anchor_dst_idx) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, _two_pass_store_anchor_to_dst_, anchor_dst_idx, VectorMode::RC_custom);
}

/**
 * @brief Restore the retained anchor from raw offset zero of the selected DST tile.
 *
 * @param anchor_dst_idx: Source tile containing the anchor.
 */
inline void llk_math_two_pass_sfpu_load_anchor_from_dst(const std::uint32_t anchor_dst_idx) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, _two_pass_load_anchor_from_dst_, anchor_dst_idx, VectorMode::RC_custom);
}

/**
 * @brief Store the retained anchor in the unused vector slot of a spilled mean-state tile.
 *
 * @param mean_dst_idx: Mean-state tile whose raw offset 4 receives the anchor.
 */
inline void llk_math_two_pass_sfpu_store_anchor_to_state_dst(const std::uint32_t mean_dst_idx) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, _two_pass_store_anchor_to_state_dst_, mean_dst_idx, VectorMode::RC_custom);
}

/**
 * @brief Restore the retained anchor from the unused vector slot of a spilled mean-state tile.
 *
 * @param mean_dst_idx: Mean-state tile whose raw offset 4 contains the anchor.
 */
inline void llk_math_two_pass_sfpu_load_anchor_from_state_dst(const std::uint32_t mean_dst_idx) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, _two_pass_load_anchor_from_state_dst_, mean_dst_idx, VectorMode::RC_custom);
}

/**
 * @brief Merge the active block with mean/M2 state in two consecutive DST tiles.
 *
 * @tparam dual_m2: Fold the secondary M2 accumulator before merging.
 * @param mean_dst_idx: First of the consecutive statistics DST tiles.
 * @param total_reciprocal_bits: FP32 bits for the reciprocal combined population.
 * @param block_n_bits: FP32 bits for the active block population.
 */
template <bool dual_m2>
inline void llk_math_two_pass_sfpu_combine_block_to_dst(
    const std::uint32_t mean_dst_idx, const std::uint32_t total_reciprocal_bits, const std::uint32_t block_n_bits) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _two_pass_combine_block_to_dst_,
        (dual_m2),
        mean_dst_idx,
        VectorMode::RC_custom,
        total_reciprocal_bits,
        block_n_bits);
}

/**
 * @brief Finalise variance and optionally the mean into row zero of two consecutive DST tiles.
 *
 * @tparam dual_m2: Fold the secondary M2 accumulator before scaling.
 * @tparam store_mean: Store the mean as well as variance; otherwise leave the mean tile unchanged.
 * @param mean_dst_idx: First of the consecutive output DST tiles.
 * @param reciprocal_bits: FP32 bit pattern for the reciprocal population count.
 */
template <bool dual_m2, bool store_mean = true>
inline void llk_math_two_pass_sfpu_store_mean_var_to_dst_row(
    const std::uint32_t mean_dst_idx, const std::uint32_t reciprocal_bits) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _two_pass_store_mean_var_to_dst_row_,
        (dual_m2, store_mean),
        mean_dst_idx,
        VectorMode::RC_custom,
        reciprocal_bits);
}

/**
 * @brief Finalise one group's mean and variance into compact raw-face slots.
 *
 * @tparam dual_m2: Fold the secondary M2 accumulator before scaling.
 * @param mean_dst_idx: First of the consecutive output DST tiles.
 * @param group_id: Four-lane group slot in the output vectors.
 * @param reciprocal_bits: FP32 bit pattern for the reciprocal population count.
 */
template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_mean_var_to_dst_raw(
    const std::uint32_t mean_dst_idx, const std::uint32_t group_id, const std::uint32_t reciprocal_bits) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _two_pass_store_mean_var_to_dst_raw_group_,
        (dual_m2),
        mean_dst_idx,
        VectorMode::RC_custom,
        group_id,
        reciprocal_bits);
}

/**
 * @brief Combine 32 equal lane populations and store one group's mean and variance in compact raw-face slots.
 *
 * Uses three consecutive DST tiles; the third is scratch and is clobbered.
 *
 * @tparam dual_m2: Fold the secondary M2 accumulator before scaling.
 * @tparam average_variance: Average the combined lane variance rather than storing its sum.
 * @param mean_dst_idx: First of the three consecutive DST tiles.
 * @param group_id: Four-lane group slot in the output vectors.
 * @param reciprocal_bits: FP32 bit pattern for each lane population's reciprocal.
 */
template <bool dual_m2, bool average_variance = true>
inline void llk_math_two_pass_sfpu_store_combined_mean_var_to_dst_raw(
    const std::uint32_t mean_dst_idx, const std::uint32_t group_id, const std::uint32_t reciprocal_bits) {
    welfords_detail::check_consecutive_dst<welfords_detail::COMBINED_STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _two_pass_store_combined_mean_var_to_dst_raw_group_,
        (dual_m2, average_variance),
        mean_dst_idx,
        VectorMode::RC_custom,
        group_id,
        reciprocal_bits);
}

/**
 * @brief Save one compact group state and restore another.
 *
 * @tparam dual_accumulator: Must be false because only the primary accumulator is preserved.
 * @param mean_dst_idx: First of the consecutive state DST tiles.
 * @param save_group_id: Group slot that receives the active state.
 * @param restore_group_id: Group slot restored into the active registers.
 */
template <bool dual_accumulator>
inline void llk_math_two_pass_sfpu_switch_group(
    const std::uint32_t mean_dst_idx, const std::uint32_t save_group_id, const std::uint32_t restore_group_id) {
    welfords_detail::check_consecutive_dst<welfords_detail::STATS_TILES>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _two_pass_switch_group_,
        (dual_accumulator),
        mean_dst_idx,
        VectorMode::RC_custom,
        save_group_id,
        restore_group_id);
}

}  // namespace ckernel
