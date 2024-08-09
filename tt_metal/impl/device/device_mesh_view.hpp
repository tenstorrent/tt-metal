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

#include "tt_metal/impl/device/device.hpp"

namespace tt::tt_metal {

// Forward declaration of DeviceMesh
class DeviceMesh;
using DeviceGrid = std::pair<int, int>;

struct Coordinate {
    std::size_t row;
    std::size_t col;
    auto operator<=>(const Coordinate&) const = default;

    // Add support for structured bindings
    template <std::size_t I>
    decltype(auto) get() const {
        if constexpr (I == 0) return row;
        else if constexpr (I == 1) return col;
        else static_assert(I < 2, "Index out of bounds for Coordinate");
    }

    friend std::ostream& operator<<(std::ostream& os, const Coordinate& coord) {
        return os << "Coord(" << coord.row << ", " << coord.col << ")";
    }
};

/**
 * @brief The DeviceMeshView class provides a view of a specific sub-region within the DeviceMesh.
 *
 * Once a DeviceMesh is initialized, DeviceMeshView allows the creation of multiple "views" on the
 * DeviceMesh, enabling more granular control over a cluster of initialized devices. This approach
 * differs from simply creating a new DeviceMesh on a subset of devices.
 *
 * DeviceMeshView serves two primary purposes:
 *
 * 1. It facilitates the creation of abstractions that define parallelization strategies, such as
 *    tensor-parallel or pipeline-parallel, by assigning views of the DeviceMesh.
 *
 * 2. It acts as a query interface for the DeviceMesh, allowing the retrieval of devices based on
 *    specific sub-regions. This is particularly useful for collective communication operations
 *    (CCL-ops), such as line all-gather, which require column or row views of the device mesh.
 */
class DeviceMeshView {
public:
    using device_pointer = Device*;
    using const_device_pointer = const Device*;
    using DeviceView = std::vector<device_pointer>;
    using DeviceViews = std::vector<std::vector<device_pointer>>;
    using CoordinateMapper = std::function<std::optional<Coordinate>(int device_id)>;

    DeviceMeshView(const DeviceMesh& mesh);
    DeviceMeshView(const DeviceMesh& mesh, Coordinate top_left, Coordinate bottom_right);
    DeviceMeshView(std::vector<device_pointer> devices, CoordinateMapper mapper);

    [[nodiscard]] device_pointer get_device(int row, int col);
    [[nodiscard]] const_device_pointer get_device(int row, int col) const;

    [[nodiscard]] const std::vector<device_pointer>& get_devices() const;

    // Get devices spanning the rectangular region defined by the top-left and bottom-right coordinates
    // devices are returned in row-major order with start/end coordinates inclusive
    [[nodiscard]] DeviceView get_devices(const Coordinate& start, const Coordinate& end);
    [[nodiscard]] DeviceView get_devices(const DeviceGrid& shape);

    [[nodiscard]] DeviceView get_devices_on_row(int row) const;
    [[nodiscard]] DeviceView get_devices_on_column(int col) const;

    [[nodiscard]] DeviceViews get_row_views() const;
    [[nodiscard]] DeviceViews get_column_views() const;

    template<typename Pred>
    [[nodiscard]] DeviceMeshView subview(Pred&& predicate) const;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] std::pair<int, int> shape() const noexcept;
    [[nodiscard]] bool contains(const Coordinate& coord) const noexcept;
    [[nodiscard]] const_device_pointer at(const Coordinate& coord) const noexcept;

    bool operator==(const DeviceMeshView& other) const;

    auto begin() const { return devices_.begin(); }
    auto end() const { return devices_.end(); }

    [[nodiscard]] std::size_t num_rows() const { return bottom_right_.row - top_left_.row + 1; }
    [[nodiscard]] std::size_t num_cols() const { return bottom_right_.col - top_left_.col + 1; }
    [[nodiscard]] std::size_t num_devices() const { return devices_.size(); }

    [[nodiscard]] Coordinate find_device(chip_id_t device_id) const;
    [[nodiscard]] chip_id_t find_device_id(const Coordinate& coord) const;

private:
    std::vector<device_pointer> devices_;
    std::unordered_map<chip_id_t, Coordinate> device_coordinates_;
    Coordinate top_left_;
    Coordinate bottom_right_;

    void initialize_from_devices(const std::vector<device_pointer>& devices, CoordinateMapper mapper);
    void validate_coordinates() const;
};

// Helper function to create a DeviceMeshView
inline DeviceMeshView make_device_mesh_view(std::vector<Device*> devices, DeviceMeshView::CoordinateMapper mapper) {
    return DeviceMeshView(std::move(devices), std::move(mapper));
}

} // namespace tt::tt_metal

// Specializations to enable structured bindings
namespace std {
    template<> struct tuple_size<tt::tt_metal::Coordinate> : std::integral_constant<std::size_t, 2> {};
    template<std::size_t I> struct tuple_element<I, tt::tt_metal::Coordinate> {
        using type = std::size_t;
    };
} // namespace std
