// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <optional>
#include <vector>

template <typename T>
std::vector<uint32_t> pack_as_bfp8_tiles(
    ttsl::Span<const T> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<float> unpack_bfp8_tiles_into_float_vec(
    ttsl::Span<const uint32_t> bfp8_tiles,
    bool row_major_output,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

// Same as above, with the exponent section padded to `l1_alignment` bytes instead of the HAL's L1
// alignment, so it does not create a MetalContext.
std::vector<float> unpack_bfp8_tiles_into_float_vec(
    ttsl::Span<const uint32_t> bfp8_tiles,
    bool row_major_output,
    bool is_exp_a,
    uint32_t l1_alignment,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
