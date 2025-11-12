// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "llk_math_welfords_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"

namespace ckernel {

inline void llk_math_welfords_sfpu_init() { _llk_math_welfords_sfpu_init_(); }

inline void llk_math_welfords_sfpu_clear_previous_mean_and_m2() { ckernel::sfpu::_clear_previous_mean_and_m2_(); }

template <uint32_t reciprocal_size>
inline void llk_math_welfords_sfpu_calculate_welfords_tile_(
    uint32_t input_dst_idx, uint32_t start_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_calculate_welfords_tile_<reciprocal_size>, input_dst_idx, start_idx, reciprocal_lut);
}

template <uint32_t reciprocal_size>
inline void llk_math_welfords_sfpu_calculate_welfords_partial_tile_(
    uint32_t input_dst_idx,
    uint32_t start_idx,
    uint32_t start_row,
    uint32_t num_rows,
    const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_calculate_welfords_partial_tile_<reciprocal_size>,
        input_dst_idx,
        start_idx,
        start_row,
        num_rows,
        reciprocal_lut);
}

inline void llk_math_welfords_sfpu_store_mean_m2_to_dst(uint32_t mean_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_store_mean_m2_to_dst_, mean_dst_idx);
}

inline void llk_math_welfords_sfpu_load_mean_m2_from_dst(uint32_t mean_dst_idx) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_load_mean_m2_from_dst_, mean_dst_idx);
}

template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_row(
    uint32_t mean_dst_idx, uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_store_mean_var_to_dst_row_<reciprocal_size>, mean_dst_idx, scale_idx, reciprocal_lut);
}

template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_raw(
    uint32_t mean_dst_idx, uint32_t scale_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_store_mean_var_to_dst_raw_<reciprocal_size>, mean_dst_idx, scale_idx, reciprocal_lut);
}

// ----------------------------------------------------------------------------
// The below functions are flavors of above 3 to use with group_id argument
// ----------------------------------------------------------------------------
inline void llk_math_welfords_sfpu_store_mean_m2_to_dst(uint32_t mean_dst_idx, uint32_t group_id) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_store_mean_m2_to_dst_group_, mean_dst_idx, group_id);
}

inline void llk_math_welfords_sfpu_load_mean_m2_from_dst(uint32_t mean_dst_idx, uint32_t group_id) {
    _llk_math_welfords_sfpu_params_(ckernel::sfpu::_load_mean_m2_from_dst_group_, mean_dst_idx, group_id);
}

template <std::size_t reciprocal_size>
inline void llk_math_welfords_sfpu_store_mean_var_to_dst_raw(
    uint32_t mean_dst_idx,
    uint32_t group_id,
    uint32_t scale_idx,
    const std::array<uint32_t, reciprocal_size>& reciprocal_lut) {
    _llk_math_welfords_sfpu_params_(
        ckernel::sfpu::_store_mean_var_to_dst_raw_group_<reciprocal_size>,
        mean_dst_idx,
        group_id,
        scale_idx,
        reciprocal_lut);
}
}  // namespace ckernel
