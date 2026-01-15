// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/span.hpp>
#include <cstddef>
#include <optional>
#include <vector>

namespace tt {
enum class DataFormat : uint8_t;
}  // namespace tt

uint32_t get_byte(uint32_t word, uint32_t index);

template <tt::DataFormat BfpFormat, bool truncate_bfp_mantissa = false>
uint8_t convert_u32_to_bfp(uint32_t input, uint32_t shared_exp, bool is_exp_a);

uint32_t convert_bfp_to_u32(tt::DataFormat bfp_format, uint8_t data, uint8_t shared_exp, bool is_exp_a);

template <tt::DataFormat BfpFormat>
[[deprecated("Use pack_as_bfp_tiles instead.")]]
std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles(
    tt::stl::Span<const float> fp32_vec,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

template <tt::DataFormat BfpFormat, typename T>
std::vector<uint32_t> pack_as_bfp_tiles(
    tt::stl::Span<const T> input_data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
