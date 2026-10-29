// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

/**
 * Copy every tile of `source` ([.., Rs, C], TILE, bf16) into `destination` ([.., R, C], same width) starting at
 * tile-row `tile_row_offset`, in place; the rest of `destination` is untouched. Returns `destination`. The copy is
 * split over all cores by tile (a plain reader/writer pair, no compute), unlike `ttnn.fill_cache` which gives
 * one core per input tile-row.
 */
ttnn::Tensor write_tile_block(const ttnn::Tensor& destination, const ttnn::Tensor& source, uint32_t tile_row_offset);

}  // namespace ttml::metal
