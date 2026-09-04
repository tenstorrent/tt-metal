// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_common_api.h"
#include "llk_sfpu_types.h"
#include "llk_math_welfords_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"

namespace ckernel {

inline void llk_math_welfords_sfpu_init() { _llk_math_welfords_sfpu_init_(); }

/**
 * @brief Configure the SFPU for two-pass statistics.
 *
 * @note Call @ref llk_math_two_pass_sfpu_clear_stats before accumulating a new population; initialisation does not
 * clear the running register state.
 */
inline void llk_math_two_pass_sfpu_init() { _llk_math_two_pass_sfpu_init_(); }

inline void llk_math_welfords_sfpu_clear_previous_mean_and_m2() { ckernel::sfpu::_clear_previous_mean_and_m2_(); }

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
    std::uint32_t input_dst_idx, std::uint32_t start_row, std::uint32_t num_rows) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_update_rows_<dual_m2>, input_dst_idx, start_row, num_rows);
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
    std::uint32_t input_dst_idx, std::uint32_t start_row, std::uint32_t num_rows) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_update_shifted_rows_<accumulate_m2, initialize_anchor, dual_accumulator>,
        input_dst_idx,
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
inline void llk_math_two_pass_sfpu_finish_shifted_mean(std::uint32_t reciprocal_bits) {
    ckernel::sfpu::_two_pass_finish_shifted_mean_<dual_sum, retain_anchor>(reciprocal_bits);
}

/** @brief Clear the active mean, shifted-sum, and M2 register state. */
inline void llk_math_two_pass_sfpu_clear_stats() { ckernel::sfpu::_two_pass_clear_stats_(); }

/**
 * @brief Spill the current mean and M2 into two consecutive DST tiles.
 *
 * @tparam dual_m2: Fold the secondary M2 accumulator before storing.
 * @param mean_dst_idx: First of the consecutive statistics DST tiles.
 */
template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_mean_m2_to_dst(std::uint32_t mean_dst_idx) {
    LLK_ASSERT(
        (mean_dst_idx + 1 < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "two-pass statistics require two consecutive DST tiles");
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_store_mean_m2_to_dst_<dual_m2>, mean_dst_idx);
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
    std::uint32_t mean_dst_idx, std::uint32_t reciprocal_bits) {
    LLK_ASSERT(
        (mean_dst_idx + 1 < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "two-pass statistics require two consecutive DST tiles");
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_store_split_mean_var_to_dst_row_<dual_m2>, mean_dst_idx, reciprocal_bits);
}

/**
 * @brief Store the retained anchor at raw offset zero of the selected DST tile.
 *
 * @param anchor_dst_idx: Destination tile for the anchor.
 */
inline void llk_math_two_pass_sfpu_store_anchor_to_dst(std::uint32_t anchor_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_store_anchor_to_dst_, anchor_dst_idx);
}

/**
 * @brief Restore the retained anchor from raw offset zero of the selected DST tile.
 *
 * @param anchor_dst_idx: Source tile containing the anchor.
 */
inline void llk_math_two_pass_sfpu_load_anchor_from_dst(std::uint32_t anchor_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_load_anchor_from_dst_, anchor_dst_idx);
}

/**
 * @brief Store the retained anchor in the unused vector slot of a spilled mean-state tile.
 *
 * @param mean_dst_idx: Mean-state tile whose raw offset 4 receives the anchor.
 */
inline void llk_math_two_pass_sfpu_store_anchor_to_state_dst(std::uint32_t mean_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_store_anchor_to_state_dst_, mean_dst_idx);
}

/**
 * @brief Restore the retained anchor from the unused vector slot of a spilled mean-state tile.
 *
 * @param mean_dst_idx: Mean-state tile whose raw offset 4 contains the anchor.
 */
inline void llk_math_two_pass_sfpu_load_anchor_from_state_dst(std::uint32_t mean_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_load_anchor_from_state_dst_, mean_dst_idx);
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
    std::uint32_t mean_dst_idx, std::uint32_t total_reciprocal_bits, std::uint32_t block_n_bits) {
    LLK_ASSERT(
        (mean_dst_idx + 1 < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "two-pass statistics require two consecutive DST tiles");
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_combine_block_to_dst_<dual_m2>, mean_dst_idx, total_reciprocal_bits, block_n_bits);
}

/**
 * @brief Finalise mean and variance into row zero of two consecutive DST tiles.
 *
 * @tparam dual_m2: Fold the secondary M2 accumulator before scaling.
 * @param mean_dst_idx: First of the consecutive output DST tiles.
 * @param reciprocal_bits: FP32 bit pattern for the reciprocal population count.
 */
template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_mean_var_to_dst_row(
    std::uint32_t mean_dst_idx, std::uint32_t reciprocal_bits) {
    LLK_ASSERT(
        (mean_dst_idx + 1 < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "two-pass statistics require two consecutive DST tiles");
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_store_mean_var_to_dst_row_<dual_m2>, mean_dst_idx, reciprocal_bits);
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
    std::uint32_t mean_dst_idx, std::uint32_t group_id, std::uint32_t reciprocal_bits) {
    LLK_ASSERT(
        (mean_dst_idx + 1 < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "two-pass statistics require two consecutive DST tiles");
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_store_mean_var_to_dst_raw_group_<dual_m2>, mean_dst_idx, group_id, reciprocal_bits);
}

/**
 * @brief Combine 32 equal lane populations and store one group's mean and variance in compact raw-face slots.
 *
 * Uses three consecutive DST tiles; the third is scratch and is clobbered.
 *
 * @tparam dual_m2: Fold the secondary M2 accumulator before scaling.
 * @param mean_dst_idx: First of the three consecutive DST tiles.
 * @param group_id: Four-lane group slot in the output vectors.
 * @param reciprocal_bits: FP32 bit pattern for each lane population's reciprocal.
 */
template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_combined_mean_var_to_dst_raw(
    std::uint32_t mean_dst_idx, std::uint32_t group_id, std::uint32_t reciprocal_bits) {
    // The implementation stores mean and variance at mean_dst_idx and mean_dst_idx + 1,
    // and clobbers mean_dst_idx + 2 as scratch while combining lane populations.
    LLK_ASSERT(
        (mean_dst_idx + 2 < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "two-pass combined statistics require three consecutive DST tiles");
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_store_combined_mean_var_to_dst_raw_group_<dual_m2>,
        mean_dst_idx,
        group_id,
        reciprocal_bits);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_welfords_sfpu_reinit(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);
    _llk_math_welfords_sfpu_reinit_<is_fp32_dest_acc_en>(num_faces, dst_format);
}

template <std::uint32_t reciprocal_size>
inline void llk_math_welfords_sfpu_calculate_welfords_tile_(
    std::uint32_t input_dst_idx,
    std::uint32_t start_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_calculate_welfords_tile_<reciprocal_size>, input_dst_idx, start_idx, reciprocal_lut);
}

template <std::uint32_t reciprocal_size>
inline void llk_math_welfords_sfpu_calculate_welfords_partial_tile_(
    std::uint32_t input_dst_idx,
    std::uint32_t start_idx,
    std::uint32_t start_row,
    std::uint32_t num_rows,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_calculate_welfords_partial_tile_<reciprocal_size>,
        input_dst_idx,
        start_idx,
        start_row,
        num_rows,
        reciprocal_lut);
}

inline void llk_math_welfords_sfpu_store_mean_m2_to_dst(std::uint32_t mean_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_store_mean_m2_to_dst_, mean_dst_idx);
}

inline void llk_math_welfords_sfpu_load_mean_m2_from_dst(std::uint32_t mean_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_load_mean_m2_from_dst_, mean_dst_idx);
}

template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_row(
    std::uint32_t mean_dst_idx,
    std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_store_mean_var_to_dst_row_<reciprocal_size>, mean_dst_idx, scale_idx, reciprocal_lut);
}

template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_raw(
    std::uint32_t mean_dst_idx,
    std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_store_mean_var_to_dst_raw_<reciprocal_size>, mean_dst_idx, scale_idx, reciprocal_lut);
}

// ----------------------------------------------------------------------------
// The below functions are flavors of above 3 to use with group_id argument
// ----------------------------------------------------------------------------
inline void llk_math_welfords_sfpu_store_mean_m2_to_dst(std::uint32_t mean_dst_idx, std::uint32_t group_id) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_store_mean_m2_to_dst_group_, mean_dst_idx, group_id);
}

inline void llk_math_welfords_sfpu_load_mean_m2_from_dst(std::uint32_t mean_dst_idx, std::uint32_t group_id) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_load_mean_m2_from_dst_group_, mean_dst_idx, group_id);
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
    std::uint32_t mean_dst_idx, std::uint32_t save_group_id, std::uint32_t restore_group_id) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_switch_group_<dual_accumulator>, mean_dst_idx, save_group_id, restore_group_id);
}

template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_raw(
    std::uint32_t mean_dst_idx,
    std::uint32_t group_id,
    std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_store_mean_var_to_dst_raw_group_<reciprocal_size>,
        mean_dst_idx,
        group_id,
        scale_idx,
        reciprocal_lut);
}
}  // namespace ckernel
