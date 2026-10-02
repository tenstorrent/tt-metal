// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::experimental {

// Same as ::unpack_bfp8_tiles_into_float_vec, with the exponent section padded to `l1_alignment`
// bytes instead of the HAL's L1 alignment, so it does not create a MetalContext.
std::vector<float> unpack_bfp8_tiles_into_float_vec(
    ttsl::Span<const uint32_t> bfp8_tiles,
    bool row_major_output,
    bool is_exp_a,
    uint32_t l1_alignment,
    const std::optional<Tile>& tile = std::nullopt);

}  // namespace tt::tt_metal::experimental
