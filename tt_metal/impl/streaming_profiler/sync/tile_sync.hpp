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
using TileClocks = std::map<std::pair<CoreType, CoreCoord>, int64_t>;

// When this process profiles its mesh devices, prepares each chip for the clock sync. It measures the chip's tile wall
// clocks against its first Tensix tile and keeps them for tile_clocks(). It also zeroes the state each active eth core
// keeps as a link sync port. A chip whose clocks this process has already measured keeps them. It must run before the
// fabric and dispatch firmware start, because it launches kernels on the Tensix, eth and DRAM cores and lets only one
// tile per chip read at a time, and because a fabric router that runs a link sync port must not read what an earlier
// process left there.
void prepare_clock_sync(std::span<Device* const> devices, ContextId context_id);

// The tile clocks prepare_clock_sync measured on `chip`, or null if it measured none there.
const TileClocks* tile_clocks(ContextId context_id, uint32_t chip);

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
