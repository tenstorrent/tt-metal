// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

// Forward declaration
enum NOC : uint8_t;

class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshCoordinate;
}

namespace tt::tt_metal::experimental::Device {

// Returns the hop distance between two logical worker coordinates on a given NOC
// This API is experimental and may evolve into a stable Device API in the future
uint32_t get_worker_noc_hop_distance(
    IDevice* device, const CoreCoord& logical_src, const CoreCoord& logical_dst, NOC noc);

// Returns the hop distance between two logical worker coordinates on a given NOC
// NOC distances may vary depending on the target device due to harvesting
// `mesh_coord` selects the device to measure on. When it maps to a device this rank does not drive
// (a submesh co-owned by several ranks), the distance is measured on an arbitrary local device
// instead: exact only when the mesh is homogeneously harvested. Throws if the mesh has no local
// device to fall back to.
// This API is experimental and may evolve into a stable Device API in the future
uint32_t get_worker_noc_hop_distance(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshCoordinate& mesh_coord,
    const CoreCoord& logical_src,
    const CoreCoord& logical_dst,
    NOC noc);

// Returns the logical worker coordinate with the fewest hops to logical_eth_core on a given NOC, and
// writes that hop count to noc_hops. The distance is measured worker -> eth core.
// Searches the compute-with-storage grid only, so the result is always a core an op can claim, and
// dispatch-reserved cores are never returned. Ties go to the lowest (y, x).
// Takes a single-device IDevice or a unit MeshDevice and throws on a larger mesh; a caller holding a
// multi-device mesh passes mesh->get_device(coord).
// This API is experimental and may evolve into a stable Device API in the future
CoreCoord get_closest_worker_to_eth_core(
    IDevice* device, const CoreCoord& logical_eth_core, NOC noc, uint32_t& noc_hops);
}  // namespace tt::tt_metal::experimental::Device
