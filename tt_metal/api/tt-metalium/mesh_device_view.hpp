// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
#include <tt-metalium/maybe_remote.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>

namespace tt::tt_metal::distributed {

// Forward declarations
class MeshDevice;
class MeshDeviceViewImpl;

/**
 * @brief The MeshDeviceView class provides a view of a specific sub-region within the MeshDevice.
 *
 * Once a MeshDevice is initialized, MeshDeviceView allows the creation of multiple "views" on the
 * MeshDevice, enabling more granular control over a cluster of initialized devices. This approach
 * differs from simply creating a new MeshDevice on a subset of devices.
 *
 * MeshDeviceView serves two primary purposes:
 *
 * 1. It facilitates the creation of abstractions that define parallelization strategies, such as
 *    tensor-parallel or pipeline-parallel, by assigning views of the MeshDevice.
 *
 * 2. It acts as a query interface for the MeshDevice, allowing the retrieval of devices based on
 *    specific sub-regions. This is particularly useful for collective communication operations
 *    (CCL-ops), such as line all-gather, which require column or row views of the device mesh.
 */

class MeshDeviceView {
public:
    // Constructors for MeshDeviceView for fully and partially local meshes.
    explicit MeshDeviceView(
        const MeshShape& shape,
        const std::vector<IDevice*>& devices,
        const std::vector<tt::tt_fabric::FabricNodeId>& fabric_node_ids);
    explicit MeshDeviceView(
        const MeshShape& shape,
        const std::vector<MaybeRemote<IDevice*>>& devices,
        const std::vector<tt::tt_fabric::FabricNodeId>& fabric_node_ids);

    // Get devices spanning the region defined by `range` in row-major order with start/end coordinates inclusive
    [[nodiscard]] std::vector<IDevice*> get_devices(const MeshCoordinateRange& range) const;
    [[nodiscard]] std::vector<IDevice*> get_devices() const;
    [[nodiscard]] std::vector<tt::tt_fabric::FabricNodeId> get_fabric_node_ids() const;
    [[nodiscard]] size_t num_devices() const;

    [[nodiscard]] const MeshShape& shape() const noexcept;
    [[nodiscard]] bool contains(const MeshCoordinate& coord) const noexcept;

    // Returns `IDevice*` instance for `coord`.
    // In multi-host context, throws if `coord` is querying a remote device.
    [[deprecated(
        "Deprecated, retrieving physical devices can fail in distributed contexts. This will be removed after "
        "28-02-2026.")]] [[nodiscard]] IDevice*
    get_device(const MeshCoordinate& coord) const;

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

    [[nodiscard]] std::vector<IDevice*> get_ring_devices() const;
    [[nodiscard]] std::vector<tt::tt_fabric::FabricNodeId> get_ring_fabric_node_ids() const;

    // Returns true if the view is fully local, i.e. all devices in the view are local.
    // Throws if the coordinate is out of bounds of this view.
    [[deprecated(
        "Deprecated, is_local should be avoided as it is likely to cause issues in distributed contexts. This will be "
        "removed after 28-02-2026.")]]
    bool is_local(const MeshCoordinate& coord) const;

    // Destructor
    ~MeshDeviceView();

    // Copy constructor and assignment
    MeshDeviceView(const MeshDeviceView&);
    MeshDeviceView& operator=(const MeshDeviceView&);

    // Move constructor and assignment
    MeshDeviceView(MeshDeviceView&&) noexcept;
    MeshDeviceView& operator=(MeshDeviceView&&) noexcept;

    const MeshDeviceViewImpl& impl() const { return *pimpl_; }
    MeshDeviceViewImpl& impl() { return *pimpl_; }

private:
    std::unique_ptr<MeshDeviceViewImpl> pimpl_;
};

}  // namespace tt::tt_metal::distributed
