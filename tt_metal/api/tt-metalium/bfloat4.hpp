// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <optional>
#include <vector>

// Set optimize_bfp=true to compare Emax and Emax-1 for each group of 16 values.
// Select the exponent with the smaller squared weight error. Requires is_exp_a=false.
// The default, optimize_bfp=false, uses the usual maximum exponent.
template <typename T>
std::vector<uint32_t> pack_as_bfp4_tiles(
    ttsl::Span<const T> data,
    bool row_major_input,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt,
    bool optimize_bfp = false);

std::vector<float> unpack_bfp4_tiles_into_float_vec(
    ttsl::Span<const uint32_t> bfp_tiles,
    bool row_major_output,
    bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
