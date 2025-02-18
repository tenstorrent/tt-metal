// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <optional>
#include <functional>

#include "device.hpp"
#include "mesh_config.hpp"

namespace tt::tt_metal::distributed {

// Forward declaration of MeshDevice
class MeshDevice;

// TODO (Issue #17477): MeshWorkload and MeshEvent currently rely on the coordinate systems
// exposed below. These must be uplifted to an ND coordinate system (DeviceCoord and DeviceRange),
// keeping things more consistent  across the stack.
// For now, since the LogicalDeviceRange concept is fundamentally identical to the CoreRange concept
// on a 2D Mesh use this definition. CoreRange contains several utility functions required
// in the MeshWorkload context.

using DeviceCoord = CoreCoord;
using LogicalDeviceRange = CoreRange;

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
    using DeviceView = std::vector<IDevice*>;
    using DeviceViews = std::vector<std::vector<IDevice*>>;
    using MeshCoordinateMapper = std::function<std::optional<MeshCoordinate>(int device_id)>;

    MeshDeviceView(const std::vector<IDevice*>& devices, const MeshShape& shape);
    MeshDeviceView(const std::vector<IDevice*>& devices, MeshCoordinate top_left, MeshCoordinate bottom_right);
    MeshDeviceView(const MeshDevice& mesh_device);
    MeshDeviceView(const std::vector<IDevice*>& devices, const MeshCoordinateMapper& mapper);

    [[nodiscard]] IDevice* get_device(size_t row, size_t col) const;

    // Get devices spanning the rectangular region defined by the top-left and bottom-right coordinates
    // devices are returned in row-major order with start/end coordinates inclusive
    [[nodiscard]] DeviceView get_devices(const MeshCoordinate& start, const MeshCoordinate& end) const;
    [[nodiscard]] DeviceView get_devices(const MeshShape& submesh_shape) const;
    [[nodiscard]] DeviceView get_devices() const;

    [[nodiscard]] DeviceView get_devices_on_row(size_t row) const;
    [[nodiscard]] DeviceView get_devices_on_column(size_t col) const;

    [[nodiscard]] DeviceViews get_row_views() const;
    [[nodiscard]] DeviceViews get_column_views() const;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] MeshShape shape() const noexcept;
    [[nodiscard]] bool contains(const MeshCoordinate& coord) const noexcept;
    [[nodiscard]] const IDevice* at(const MeshCoordinate& coord) const noexcept;

    bool operator==(const MeshDeviceView& other) const;

    auto begin() const { return devices_.begin(); }
    auto end() const { return devices_.end(); }

    // [[nodiscard]] size_t num_rows() const { return bottom_right_.row - top_left_.row + 1; }
    // [[nodiscard]] size_t num_cols() const { return bottom_right_.col - top_left_.col + 1; }
    [[nodiscard]] size_t num_devices() const { return devices_.size(); }

    [[nodiscard]] bool contains_device(chip_id_t device_id) const;
    [[nodiscard]] MeshCoordinate find_device(chip_id_t device_id) const;
    [[nodiscard]] chip_id_t find_device_id(const MeshCoordinate& coord) const;

    // These utility methods linearize the set of devices in a mesh into a line or ring.
    // Linearizing a mesh into a line asserts the condition that device[i-1] is connected to device[i].
    // Linearizing a mesh into a ring asserts the condition that device[i-1] is connected to device[i] and device[0] is
    // connected to device[-1].
    //
    // Given a starting coordinate, get the coordinates of a line of devices where device[i-1] is connected to device[i]
    // The current support only provides left-to-right and right-to-left snaking of the line.
    [[nodiscard]] static std::vector<MeshCoordinate> get_line_coordinates(
        size_t length, const MeshCoordinate& offset, size_t num_rows, size_t num_cols);
    [[nodiscard]] std::vector<MeshCoordinate> get_ring_coordinates(
        const MeshShape& ring_shape, const MeshCoordinate& offset, size_t num_rows, size_t num_cols) const;
    [[nodiscard]] std::vector<IDevice*> get_ring_devices() const;
    [[nodiscard]] std::vector<IDevice*> get_line_devices() const;

private:
    std::vector<IDevice*> devices_;
    std::unordered_map<chip_id_t, MeshCoordinate> device_coordinates_;
    MeshCoordinate top_left_;
    MeshCoordinate bottom_right_;

    void initialize_from_devices(const std::vector<IDevice*>& devices, const MeshCoordinateMapper& mapper);
    void validate_coordinates() const;
};

// Helper function to create a MeshDeviceView
inline MeshDeviceView make_mesh_device_view(
    std::vector<IDevice*> devices, MeshDeviceView::MeshCoordinateMapper mapper) {
    return MeshDeviceView(std::move(devices), std::move(mapper));
}

}  // namespace tt::tt_metal::distributed
