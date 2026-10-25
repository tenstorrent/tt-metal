// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/context/mesh_device_properties.hpp>

#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::experimental {

MetalDeviceProperties get_device_properties(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& coord) {
    const auto& shape = mesh_device.shape();
    // MeshCoordinateRange::contains asserts on a rank mismatch, so the rank is checked first.
    TT_FATAL(
        coord.dims() == shape.dims(),
        "Mesh coordinate {} has {} dimensions, but mesh shape {} has {}",
        coord,
        coord.dims(),
        shape,
        shape.dims());
    TT_FATAL(
        distributed::MeshCoordinateRange(shape).contains(coord),
        "Mesh coordinate {} is outside mesh shape {}",
        coord,
        shape);
    return get_device_properties(mesh_device);
}

MetalDeviceProperties get_device_properties(const distributed::MeshDevice& mesh_device) {
    return MetalDeviceProperties(mesh_device.impl().metal_env());
}

}  // namespace tt::tt_metal::experimental
