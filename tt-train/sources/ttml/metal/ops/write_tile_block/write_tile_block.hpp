// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

/**
 * Copy every tile of `source` ([.., Rs, Cs], TILE, bf16) into `destination` ([.., R, C]) starting at tile-row
 * `tile_row_offset` and tile-column `tile_col_offset`, in place; the rest of `destination` is untouched. Returns
 * `destination`. The copy is split over all cores by tile (a plain reader/writer pair, no compute), unlike
 * `ttnn.fill_cache` which gives one core per input tile-row. A full-width source (Cs == C, the LoRA tail rows of a
 * fused weight) is one contiguous tile range; a narrower one (the tail columns of the transposed forward copy,
 * iteration 36) lands tile by tile.
 */
ttnn::Tensor write_tile_block(
    const ttnn::Tensor& destination,
    const ttnn::Tensor& source,
    uint32_t tile_row_offset,
    uint32_t tile_col_offset = 0U);

}  // namespace ttml::metal
