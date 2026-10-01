// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>
#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>
#include <utility>

#include "ttnn/tensor/tensor.hpp"

namespace ttml::metal::ops::layernorm_common {

inline tt::tt_metal::Shape stats_logical_shape(const ttnn::Tensor& input) {
    auto shape = input.logical_shape();
    shape[-1] = 1U;
    return shape;
}

inline tt::tt_metal::Shape stats_padded_shape(const ttnn::Tensor& input) {
    auto shape = input.padded_shape();
    shape[-1] = tt::constants::TILE_WIDTH;
    return shape;
}

inline tt::tt_metal::TensorSpec stats_tensor_spec(const ttnn::Tensor& input) {
    const auto& input_layout = input.tensor_spec().tensor_layout();
    const auto& input_alignment = input_layout.get_alignment();
    ttsl::SmallVector<uint32_t> stats_alignment(input_alignment.cbegin(), input_alignment.cend());
    stats_alignment.back() = tt::constants::TILE_WIDTH;

    return tt::tt_metal::TensorSpec(
        stats_logical_shape(input),
        tt::tt_metal::TensorLayout(
            input.dtype(),
            input_layout.get_page_config(),
            input.memory_config(),
            tt::tt_metal::Alignment(std::move(stats_alignment))));
}

inline void validate_stats_geometry(const ttnn::Tensor& stats, const ttnn::Tensor& input, const std::string_view name) {
    const auto expected_logical_shape = stats_logical_shape(input);
    const auto expected_padded_shape = stats_padded_shape(input);
    TT_FATAL(
        stats.logical_shape() == expected_logical_shape && stats.padded_shape() == expected_padded_shape,
        "{} tensor must have logical shape {} and padded shape {}. Got logical shape {} and padded shape {}",
        name,
        expected_logical_shape,
        expected_padded_shape,
        stats.logical_shape(),
        stats.padded_shape());
}

}  // namespace ttml::metal::ops::layernorm_common
