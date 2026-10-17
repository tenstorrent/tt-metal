// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <utility>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "impl/context/context_types.hpp"

namespace tt::tt_metal {
class Device;

namespace streaming_profiler {

// Each tile's wall tick minus the chip's first Tensix tile's at the same instant, by core type and logical coordinate.
using TileClockOffsets = std::map<std::pair<CoreType, CoreCoord>, int64_t>;

// When this process profiles its mesh devices, measures each chip's tile wall clocks against its first Tensix tile and
// keeps the offsets for tile_clock_offsets(). A chip this process has already measured keeps its offsets. It must run
// before the fabric and dispatch firmware start, because it launches kernels on the Tensix, eth and DRAM cores and lets
// only one tile per chip read at a time.
void measure_tile_clock_offsets(std::span<Device* const> devices, ContextId context_id);

// The offsets measure_tile_clock_offsets measured on `chip`, or null if it measured none there.
const TileClockOffsets* tile_clock_offsets(ContextId context_id, uint32_t chip);

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
