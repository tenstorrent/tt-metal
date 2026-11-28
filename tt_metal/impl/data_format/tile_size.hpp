// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/data_formats/tile.hpp>
#include <tt-metalium/data_formats/data_format.hpp>

namespace tt::tt_metal::detail {

// Returns the tile storage size in bytes for the supplied L1 alignment.
// precondition: l1_alignment is a nonzero alignment in bytes.
uint32_t tile_size_bytes(const Tile& tile, DataFormat format, uint32_t l1_alignment);

}  // namespace tt::tt_metal::detail
