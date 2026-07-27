// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
// UMD: re-exports CoreType (used in get_physical_core_from_logical_core).
#include <umd/device/types/core_coordinates.hpp>

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
// This API is experimental and may evolve into a stable Device API in the future
uint32_t get_worker_noc_hop_distance(
    distributed::MeshDevice* mesh_device,
    const distributed::MeshCoordinate& mesh_coord,
    const CoreCoord& logical_src,
    const CoreCoord& logical_dst,
    NOC noc);

// Returns the PHYSICAL (noc0) coordinate of a logical core of the given type.
// Physical coords are the only space in which cores of DIFFERENT types (e.g. an eth core and a
// worker core) can be compared geometrically: virtual/translated coords put eth and tensix in
// disjoint, unrelated ranges (on Blackhole an eth core's translated x is its channel id), so
// "is this worker in the same column as that eth core?" is only answerable here.
// This API is experimental and may evolve into a stable Device API in the future.
CoreCoord get_physical_core_from_logical_core(
    IDevice* device, const CoreCoord& logical_core, const tt::CoreType& core_type);
}  // namespace tt::tt_metal::experimental::Device
