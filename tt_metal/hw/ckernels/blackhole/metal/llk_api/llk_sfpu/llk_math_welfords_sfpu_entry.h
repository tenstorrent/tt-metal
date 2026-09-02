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

inline void llk_math_two_pass_sfpu_init() { _llk_math_two_pass_sfpu_init_(); }

inline void llk_math_welfords_sfpu_clear_previous_mean_and_m2() { ckernel::sfpu::_clear_previous_mean_and_m2_(); }

template <bool accumulate_m2, bool dual_m2>
inline void llk_math_two_pass_sfpu_update_rows(
    std::uint32_t input_dst_idx, std::uint32_t start_row, std::uint32_t num_rows) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_update_rows_<accumulate_m2, dual_m2>, input_dst_idx, start_row, num_rows);
}

template <bool accumulate_m2, bool initialize_anchor, bool dual_accumulator>
inline void llk_math_two_pass_sfpu_update_shifted_rows(
    std::uint32_t input_dst_idx, std::uint32_t start_row, std::uint32_t num_rows) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_update_shifted_rows_<accumulate_m2, initialize_anchor, dual_accumulator>,
        input_dst_idx,
        start_row,
        num_rows);
}

template <bool dual_sum, bool retain_anchor = false>
inline void llk_math_two_pass_sfpu_finish_shifted_mean(std::uint32_t reciprocal_bits) {
    ckernel::sfpu::_two_pass_finish_shifted_mean_<dual_sum, retain_anchor>(reciprocal_bits);
}

inline void llk_math_two_pass_sfpu_clear_stats() { ckernel::sfpu::_two_pass_clear_stats_(); }

template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_mean_m2_to_dst(std::uint32_t mean_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_store_mean_m2_to_dst_<dual_m2>, mean_dst_idx);
}

template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_split_mean_var_to_dst_row(
    std::uint32_t mean_dst_idx, std::uint32_t reciprocal_bits) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_store_split_mean_var_to_dst_row_<dual_m2>, mean_dst_idx, reciprocal_bits);
}

inline void llk_math_two_pass_sfpu_store_anchor_to_dst(std::uint32_t anchor_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_store_anchor_to_dst_, anchor_dst_idx);
}

inline void llk_math_two_pass_sfpu_load_anchor_from_dst(std::uint32_t anchor_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_load_anchor_from_dst_, anchor_dst_idx);
}

inline void llk_math_two_pass_sfpu_store_anchor_to_state_dst(std::uint32_t mean_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_store_anchor_to_state_dst_, mean_dst_idx);
}

inline void llk_math_two_pass_sfpu_load_anchor_from_state_dst(std::uint32_t mean_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_load_anchor_from_state_dst_, mean_dst_idx);
}

template <bool dual_m2>
inline void llk_math_two_pass_sfpu_combine_block_to_dst(
    std::uint32_t mean_dst_idx, std::uint32_t total_reciprocal_bits, std::uint32_t block_n_bits) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_combine_block_to_dst_<dual_m2>, mean_dst_idx, total_reciprocal_bits, block_n_bits);
}

template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_mean_var_to_dst_row(
    std::uint32_t mean_dst_idx, std::uint32_t reciprocal_bits) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_store_mean_var_to_dst_row_<dual_m2>, mean_dst_idx, reciprocal_bits);
}

template <bool dual_m2>
inline void llk_math_two_pass_sfpu_store_mean_var_to_dst_raw(
    std::uint32_t mean_dst_idx, std::uint32_t group_id, std::uint32_t reciprocal_bits) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_two_pass_store_mean_var_to_dst_raw_group_<dual_m2>, mean_dst_idx, group_id, reciprocal_bits);
}

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
