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

// Returns the virtual NoC coordinate of a logical worker core on the device at `mesh_coord`.
// Unlike MeshDevice::worker_core_from_logical_core, this does not require every device in the mesh
// to share a logical-to-virtual mapping, so it is exact on a heterogeneously harvested mesh.
// `mesh_coord` must name a device this rank drives: the logical-to-virtual mapping comes from that
// chip's SoC descriptor, which the cluster holds only for local devices. Throws otherwise.
// This API is experimental and may evolve into a stable Device API in the future
CoreCoord worker_core_from_logical_core(
    distributed::MeshDevice* mesh_device, const distributed::MeshCoordinate& mesh_coord, const CoreCoord& logical_core);
}  // namespace tt::tt_metal::experimental::Device
