// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <optional>
#include <functional>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/maybe_remote.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>

namespace tt::tt_metal::distributed {

// Forward declaration of MeshDevice
class MeshDevice;

/**
 * @brief The MeshDeviceViewImpl class provides the implementation for MeshDeviceView.
 *
 * This class replicates the full interface of MeshDeviceView and will be used
 * as the implementation class in the PIMPL pattern.
 */

class MeshDeviceViewImpl {
public:
    // Constructors for MeshDeviceViewImpl for fully and partially local meshes.
    explicit MeshDeviceViewImpl(
        const MeshShape& shape,
        const std::vector<IDevice*>& devices,
        const std::vector<tt::tt_fabric::FabricNodeId>& fabric_node_ids);
    explicit MeshDeviceViewImpl(
        const MeshShape& shape,
        const std::vector<MaybeRemote<IDevice*>>& devices,
        const std::vector<tt::tt_fabric::FabricNodeId>& fabric_node_ids);

    // Get devices spanning the region defined by `range` in row-major order with start/end coordinates inclusive
    [[nodiscard]] std::vector<IDevice*> get_devices(const MeshCoordinateRange& range) const;
    [[nodiscard]] std::vector<IDevice*> get_devices() const;
    [[nodiscard]] std::vector<tt::tt_fabric::FabricNodeId> get_fabric_node_ids(const MeshCoordinateRange& range) const;
    [[nodiscard]] std::vector<tt::tt_fabric::FabricNodeId> get_fabric_node_ids() const;
    [[nodiscard]] size_t num_devices() const;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] const MeshShape& shape() const noexcept;
    [[nodiscard]] tt::tt_fabric::MeshId mesh_id() const noexcept;
    [[nodiscard]] bool contains(const MeshCoordinate& coord) const noexcept;

    // Returns `IDevice*` instance for `coord`.
    // In multi-host context, throws if `coord` is querying a remote device.
    [[nodiscard]] IDevice* get_device(const MeshCoordinate& coord) const;

    // Returns `tt::tt_fabric::FabricNodeId` for `coord`.
    // In multi-host context, fabric node IDs are always available, even for remote devices.
    [[nodiscard]] tt::tt_fabric::FabricNodeId get_fabric_node_id(const MeshCoordinate& coord) const;

    auto begin() const { return devices_.values().begin(); }
    auto end() const { return devices_.values().end(); }

    // Throws if no device corresponds to `device_id`.
    [[nodiscard]] MeshCoordinate find_device(ChipId device_id) const;

    // TODO: #17477 - Remove the methods that assume 2D mesh.
    [[nodiscard]] bool is_mesh_2d() const;
    [[nodiscard]] size_t num_rows() const;
    [[nodiscard]] size_t num_cols() const;
    [[nodiscard]] std::vector<IDevice*> get_devices_on_row(size_t row) const;
    [[nodiscard]] std::vector<IDevice*> get_devices_on_column(size_t col) const;
    [[nodiscard]] std::vector<tt::tt_fabric::FabricNodeId> get_fabric_node_ids_on_row(size_t row) const;
    [[nodiscard]] std::vector<tt::tt_fabric::FabricNodeId> get_fabric_node_ids_on_column(size_t col) const;

    // These utility methods linearize the set of devices in a mesh into a line or ring.
    // Linearizing a mesh into a line asserts the condition that device[i-1] is connected to device[i].
    // Linearizing a mesh into a ring asserts the condition that device[i-1] is connected to device[i] and device[0] is
    // connected to device[-1].
    //
    // Given a starting coordinate, get the coordinates of a line of devices where device[i-1] is connected to device[i]
    // The current support only provides left-to-right and right-to-left snaking of the line.
    //
    // Important: these utilities currently only support 2D meshes.
    // TODO: #17477 - Remove the methods that assume 2D mesh.
    [[nodiscard]] static std::vector<MeshCoordinate> get_line_coordinates(
        size_t length, const Shape2D& mesh_shape, const Shape2D& mesh_offset);
    [[nodiscard]] std::vector<MeshCoordinate> get_line_coordinates() const;
    [[nodiscard]] static std::vector<MeshCoordinate> get_ring_coordinates(
        const Shape2D& ring_shape, const Shape2D& mesh_shape);
    [[nodiscard]] std::vector<MeshCoordinate> get_ring_coordinates() const;
    [[nodiscard]] std::vector<IDevice*> get_ring_devices() const;
    [[nodiscard]] std::vector<IDevice*> get_line_devices() const;
    [[nodiscard]] std::vector<tt::tt_fabric::FabricNodeId> get_ring_fabric_node_ids() const;
    [[nodiscard]] std::vector<tt::tt_fabric::FabricNodeId> get_line_fabric_node_ids() const;

    // Returns true if the view is fully local, i.e. all devices in the view are local.
    // Throws if the coordinate is out of bounds of this view.
    bool is_local(const MeshCoordinate& coord) const;

    // Returns the coordinate range of all local devices in this view.
    // The range is a bounding box that encompasses all local coordinates.
    MeshCoordinateRange get_local_mesh_coord_range() const;

private:
    DistributedMeshContainer<IDevice*> devices_;
    MeshContainer<tt::tt_fabric::FabricNodeId> fabric_node_ids_;
    tt::tt_fabric::MeshId mesh_id_;

    std::unordered_map<ChipId, MeshCoordinate> device_coordinates_;

    // Set if the view is 2D to enable row/col APIs, otherwise nullopt.
    // TODO: #17477 - Remove this?
    std::optional<Shape2D> shape_2d_;
};

}  // namespace tt::tt_metal::distributed
