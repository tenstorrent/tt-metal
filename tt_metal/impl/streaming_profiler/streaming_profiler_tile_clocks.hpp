// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The chip's tile clocks. Every tile keeps its own wall clock on the one AICLK, a fixed whole number of ticks from
// every other tile's while the chip is up; the Tensix clocks halt while the chip idles between sessions and the eth
// and DRAM clocks do not, so that number runs to seconds. Before the fabric and dispatch firmware come up, every tile
// with a RISC (Tensix, eth, DRAM) reads the wall clock of every tile in its row and column over the NoC: NoC 0
// towards higher coordinates, NoC 1 towards lower, so a pair's two readings cross the same links in opposite
// directions and half their difference is the pair's offset with nothing about the path in it. The pairs are
// combined over the grid by least squares; the residuals are the loop closures.
#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "impl/context/context_types.hpp"
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {
class IDevice;
}

namespace tt::tt_metal::streaming_profiler {

struct TileClock {
    CoreType type;  // WORKER, ETH or DRAM
    CoreCoord logical, virt, phys;
    int64_t offset;  // this tile's wall tick minus the chip's first tile's, at the same instant
};

struct TileClocks {
    std::vector<TileClock> tiles;
    // A DRAM tile's reading of a tile in its row at bring-up, 2 * (tile wall - bracket midpoint) in ticks: the drift
    // check at capture end takes the same reading again.
    struct RowReading {
        uint32_t reader, tile;
        int32_t median2;
    };
    std::vector<RowReading> row_readings;
    const TileClock* find(CoreType type, const CoreCoord& logical) const;
};

// Measures every tile of the device and files the result with the Service; a device already measured is left alone.
// MUST run before the fabric and dispatch firmware: every Tensix, eth and DRAM core is launched on, one tile reading
// at a time.
void measure_tile_clocks(IDevice* device, ContextId context_id);

// The DRAM tiles read their rows again and the largest change since bring-up is logged. Nothing of ours may be on
// the DRAM cores.
void check_tile_clock_drift(IDevice* device, ContextId context_id);

}  // namespace tt::tt_metal::streaming_profiler
