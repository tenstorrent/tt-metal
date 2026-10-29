// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

// Both overloads return the same view as MetalEnv::get_device_properties() of the MetalEnv the mesh was opened on, so
// the result is valid while that MetalEnv is alive, which already outlives every MeshDevice bound to it.

/// @return Hardware properties of the device at `coord`. Coordinates are local to `mesh_device`, so a submesh is
/// addressed from its own origin. Throws if `coord` is not within the mesh shape. For a coordinate owned by a remote
/// host, the answer relies on the mesh being homogeneous.
MetalDeviceProperties get_device_properties(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& coord);

/// @return Hardware properties common to every device in `mesh_device`. Meshes are homogeneous, so this always
/// succeeds; should heterogeneous meshes ever be supported, it throws for them rather than picking one device.
MetalDeviceProperties get_device_properties(const distributed::MeshDevice& mesh_device);

}  // namespace experimental

}  // namespace tt::tt_metal
