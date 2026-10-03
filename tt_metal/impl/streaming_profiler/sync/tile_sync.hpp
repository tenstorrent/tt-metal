// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "impl/context/context_types.hpp"

namespace tt::tt_metal {
class Device;

namespace streaming_profiler {

struct TileClock {
    CoreType type;
    CoreCoord logical;
    int64_t offset;  // this tile's wall tick minus the chip's first tile's, at the same instant
};

struct TileClocks {
    std::vector<TileClock> tiles;
    int64_t offset(CoreType type, const CoreCoord& logical) const;
};

// Every tile's wall clock runs on the one AICLK, so each stays a fixed whole number of ticks from every other while the
// chip is up. That number can run to seconds, because the Tensix clocks halt while the chip idles between sessions and
// the eth and DRAM clocks don't.
// It has to run before the fabric and dispatch firmware start, since it launches on every Tensix, eth and DRAM core and
// lets only one tile per chip read at a time. A chip whose clocks this process already measured keeps them.
void measure_tile_clocks(std::span<Device* const> devices, ContextId context_id);

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
