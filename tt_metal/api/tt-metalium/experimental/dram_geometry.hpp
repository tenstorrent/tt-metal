// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental host-side geometry queries for NoC-aware placement: where each DRAM bank's tiles sit on the NOC0 grid
// and which NoC each tile is the bank's endpoint for, and where a worker core sits on the NOC0 grid, per chip
// (harvesting moves both). Host only: they also answer on a mock cluster.

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshCoordinate;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental {

// One DRAM tile of a bank, on the device at a given mesh coordinate.
struct DramBankTile {
    // {bank, index}: index 0 is the bank's NOC0 worker endpoint, 1 its NOC1 worker endpoint, then the bank's other
    // tiles.
    CoreCoord logical_core;
    // NoC address of the tile (translated coordinates).
    CoreCoord virtual_core;
    // Position on the NOC0 grid (physical), for hop distances.
    CoreCoord noc0_core;
    // Bit n set: the tile is a DRAM view's NOC n endpoint (its NOC n NIU forwards the view's DRAM accesses). 0: the
    // tile is no view's endpoint.
    uint8_t endpoint_noc_mask = 0;
};

// Every DRAM tile of `bank` (0 .. dram_grid_size().x - 1), in logical-index order. `mesh_coord` must name a device
// this rank drives.
std::vector<DramBankTile> GetDramBankTiles(
    distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& mesh_coord, uint32_t bank);

// The NOC0 grid (physical) position of a logical worker core, on the device at `mesh_coord` (a device this rank
// drives): pairs with DramBankTile::noc0_core for hop distances, and is what a worker reads as its own NoC
// coordinates.
CoreCoord GetWorkerNoc0Core(
    distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& mesh_coord, const CoreCoord& logical_core);

}  // namespace tt::tt_metal::experimental
