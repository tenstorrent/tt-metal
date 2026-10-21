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

template <std::uint32_t NUM_TILES>
inline void check_consecutive_dst(const std::uint32_t first_dst_idx) {
    constexpr std::uint32_t max_dest_tiles =
        trisc::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, trisc::DstTileShape::Tile32x32>();
    LLK_ASSERT(first_dst_idx + NUM_TILES <= max_dest_tiles, "two-pass statistics: the Dest tiles must fit in Dest");
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
        (false /* PARTIAL_TILE */, reciprocal_size),
        input_dst_idx,
        VectorMode::RC_custom,
        start_idx,
        reciprocal_lut,
        0u /* start_row */,
        static_cast<std::uint32_t>(TILE_R_DIM) /* num_rows */);
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
        (true /* PARTIAL_TILE */, reciprocal_size),
        input_dst_idx,
        VectorMode::RC_custom,
        start_idx,
        reciprocal_lut,
        start_row,
        num_rows);
}

inline void llk_math_welfords_sfpu_store_mean_m2_to_dst(const std::uint32_t mean_dst_idx) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_store_mean_m2_to_dst,
        (false),
        mean_dst_idx,
        VectorMode::RC_custom,
        0u /* group_id */);
}

inline void llk_math_welfords_sfpu_store_mean_m2_to_dst(
    const std::uint32_t mean_dst_idx, const std::uint32_t group_id) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_store_mean_m2_to_dst,
        (true),
        mean_dst_idx,
        VectorMode::RC_custom,
        group_id);
}

inline void llk_math_welfords_sfpu_load_mean_m2_from_dst(const std::uint32_t mean_dst_idx) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_load_mean_m2_from_dst,
        (false),
        mean_dst_idx,
        VectorMode::RC_custom,
        0u /* group_id */);
}

inline void llk_math_welfords_sfpu_load_mean_m2_from_dst(
    const std::uint32_t mean_dst_idx, const std::uint32_t group_id) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_load_mean_m2_from_dst,
        (true),
        mean_dst_idx,
        VectorMode::RC_custom,
        group_id);
}

template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_row(
    const std::uint32_t mean_dst_idx,
    const std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_store_mean_var_to_dst,
        (sfpu::WelfordsOutputLayout::Row, false /* GROUPED */, reciprocal_size),
        mean_dst_idx,
        VectorMode::RC_custom,
        scale_idx,
        reciprocal_lut,
        0u /* group_id */);
}

template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_raw(
    const std::uint32_t mean_dst_idx,
    const std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_store_mean_var_to_dst,
        (sfpu::WelfordsOutputLayout::Face, false /* GROUPED */, reciprocal_size),
        mean_dst_idx,
        VectorMode::RC_custom,
        scale_idx,
        reciprocal_lut,
        0u /* group_id */);
}

template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_raw(
    const std::uint32_t mean_dst_idx,
    const std::uint32_t group_id,
    const std::uint32_t scale_idx,
    const std::array<std::uint32_t, reciprocal_size>& reciprocal_lut) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        welfords_store_mean_var_to_dst,
        (sfpu::WelfordsOutputLayout::Face, true /* GROUPED */, reciprocal_size),
        mean_dst_idx,
        VectorMode::RC_custom,
        scale_idx,
        reciprocal_lut,
        group_id);
}

// ---- Shifted two-pass statistics ------------------------------------------------------------------

inline void llk_math_two_pass_sfpu_init() { llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(); }

inline void llk_math_two_pass_sfpu_clear_stats() { sfpu::_two_pass_clear_stats_(); }

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

template <bool dual_sum, bool retain_anchor = false>
inline void llk_math_two_pass_sfpu_finish_shifted_mean(const std::uint32_t reciprocal_bits) {
    sfpu::_two_pass_finish_shifted_mean_<dual_sum, retain_anchor>(reciprocal_bits);
}

template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_mean_m2_to_dst(const std::uint32_t mean_dst_idx) {
    welfords_detail::check_consecutive_dst<2>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _two_pass_store_mean_m2_to_dst_, (dual_m2), mean_dst_idx, VectorMode::RC_custom);
}

template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_split_mean_var_to_dst_row(
    const std::uint32_t mean_dst_idx, const std::uint32_t reciprocal_bits) {
    welfords_detail::check_consecutive_dst<2>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _two_pass_store_split_mean_var_to_dst_row_,
        (dual_m2),
        mean_dst_idx,
        VectorMode::RC_custom,
        reciprocal_bits);
}

inline void llk_math_two_pass_sfpu_store_anchor_to_dst(const std::uint32_t anchor_dst_idx) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, _two_pass_store_anchor_to_dst_, anchor_dst_idx, VectorMode::RC_custom);
}

inline void llk_math_two_pass_sfpu_load_anchor_from_dst(const std::uint32_t anchor_dst_idx) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, _two_pass_load_anchor_from_dst_, anchor_dst_idx, VectorMode::RC_custom);
}

inline void llk_math_two_pass_sfpu_store_anchor_to_state_dst(const std::uint32_t mean_dst_idx) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, _two_pass_store_anchor_to_state_dst_, mean_dst_idx, VectorMode::RC_custom);
}

inline void llk_math_two_pass_sfpu_load_anchor_from_state_dst(const std::uint32_t mean_dst_idx) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, _two_pass_load_anchor_from_state_dst_, mean_dst_idx, VectorMode::RC_custom);
}

template <bool dual_m2>
inline void llk_math_two_pass_sfpu_combine_block_to_dst(
    const std::uint32_t mean_dst_idx, const std::uint32_t total_reciprocal_bits, const std::uint32_t block_n_bits) {
    welfords_detail::check_consecutive_dst<2>(mean_dst_idx);
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

template <bool dual_m2, bool store_mean = true>
inline void llk_math_two_pass_sfpu_store_mean_var_to_dst_row(
    const std::uint32_t mean_dst_idx, const std::uint32_t reciprocal_bits) {
    welfords_detail::check_consecutive_dst<2>(mean_dst_idx);
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _two_pass_store_mean_var_to_dst_row_,
        (dual_m2, store_mean),
        mean_dst_idx,
        VectorMode::RC_custom,
        reciprocal_bits);
}

template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_mean_var_to_dst_raw(
    const std::uint32_t mean_dst_idx, const std::uint32_t group_id, const std::uint32_t reciprocal_bits) {
    welfords_detail::check_consecutive_dst<2>(mean_dst_idx);
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

template <bool dual_m2, bool average_variance = true>
inline void llk_math_two_pass_sfpu_store_combined_mean_var_to_dst_raw(
    const std::uint32_t mean_dst_idx, const std::uint32_t group_id, const std::uint32_t reciprocal_bits) {
    welfords_detail::check_consecutive_dst<3>(mean_dst_idx);
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

template <bool dual_accumulator>
inline void llk_math_two_pass_sfpu_switch_group(
    const std::uint32_t mean_dst_idx, const std::uint32_t save_group_id, const std::uint32_t restore_group_id) {
    welfords_detail::check_consecutive_dst<2>(mean_dst_idx);
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
