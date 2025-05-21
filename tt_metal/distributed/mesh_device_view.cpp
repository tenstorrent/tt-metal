// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <mesh_device.hpp>
#include <mesh_device_view.hpp>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "assert.hpp"
#include "device.hpp"
#include "mesh_config.hpp"
#include "mesh_coord.hpp"
#include "shape2d.hpp"
#include "shape_base.hpp"

namespace tt::tt_metal::distributed {
namespace {

std::vector<IDevice*> get_devices_from_coordinates(
    const MeshDeviceView& mesh, const std::vector<MeshCoordinate>& coords) {
    std::vector<IDevice*> devices;
    for (const auto& coord : coords) {
        if (auto device = mesh.get_device(coord)) {
            devices.push_back(device);
        }
    }
    return devices;
}

}  // namespace

MeshDeviceView::MeshDeviceView(const MeshContainer<IDevice*>& devices) : devices_(devices) {
    if (devices_.shape().dims() == 2) {
        shape_2d_ = Shape2D(devices_.shape()[0], devices_.shape()[1]);
    }
    for (const auto& [coord, device] : devices_) {
        device_coordinates_.emplace(device->id(), coord);
    }
}

MeshDeviceView::MeshDeviceView(const MeshDevice& mesh_device) :
    MeshDeviceView(MeshContainer<IDevice*>(MeshShape(mesh_device.shape()), mesh_device.get_devices())) {}

MeshDeviceView::DeviceView MeshDeviceView::get_devices(const MeshCoordinateRange& range) const {
    DeviceView devices_in_region;
    for (const auto& coord : range) {
        devices_in_region.push_back(devices_.at(coord));
    }
    return devices_in_region;
}

MeshDeviceView::DeviceView MeshDeviceView::get_devices(const MeshShape& submesh_shape) const {
    return get_devices(MeshCoordinateRange(submesh_shape));
}

std::vector<IDevice*> MeshDeviceView::get_devices_on_row(size_t row) const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceView is not 2D!");
    TT_FATAL(row < shape_2d_->height(), "Row index out of bounds: {}", row);
    std::vector<IDevice*> row_devices;
    for (int col = 0; col < shape_2d_->width(); ++col) {
        row_devices.push_back(devices_.at(MeshCoordinate(row, col)));
    }
    return row_devices;
}

std::vector<IDevice*> MeshDeviceView::get_devices_on_column(size_t col) const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceView is not 2D!");
    TT_FATAL(col < shape_2d_->width(), "Column index out of bounds: {}", col);
    std::vector<IDevice*> col_devices;
    for (int row = 0; row < shape_2d_->height(); ++row) {
        col_devices.push_back(devices_.at(MeshCoordinate(row, col)));
    }
    return col_devices;
}

std::vector<std::vector<IDevice*>> MeshDeviceView::get_row_views() const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceView is not 2D!");
    std::vector<std::vector<IDevice*>> row_views;
    for (size_t row = 0; row < shape_2d_->height(); ++row) {
        row_views.push_back(get_devices_on_row(row));
    }
    return row_views;
}

std::vector<std::vector<IDevice*>> MeshDeviceView::get_column_views() const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceView is not 2D!");
    std::vector<std::vector<IDevice*>> column_views;
    for (size_t col = 0; col < shape_2d_->width(); ++col) {
        column_views.push_back(get_devices_on_column(col));
    }
    return column_views;
}

bool MeshDeviceView::empty() const noexcept { return devices_.shape().mesh_size() == 0; }
size_t MeshDeviceView::size() const noexcept { return devices_.shape().mesh_size(); }
const MeshShape& MeshDeviceView::shape() const noexcept { return devices_.shape(); }

bool MeshDeviceView::contains(const MeshCoordinate& coord) const noexcept {
    return devices_.coord_range().contains(coord);
}

IDevice* MeshDeviceView::get_device(const MeshCoordinate& coord) const {
    return contains(coord) ? devices_.at(coord) : nullptr;
}
const IDevice* MeshDeviceView::at(const MeshCoordinate& coord) const noexcept {
    return contains(coord) ? devices_.at(coord) : nullptr;
}

bool MeshDeviceView::operator==(const MeshDeviceView& other) const {
    return devices_ == other.devices_ && device_coordinates_ == other.device_coordinates_ &&
           shape_2d_ == other.shape_2d_;
}

size_t MeshDeviceView::num_rows() const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceView is not 2D!");
    return shape_2d_->height();
}
size_t MeshDeviceView::num_cols() const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceView is not 2D!");
    return shape_2d_->width();
}
size_t MeshDeviceView::num_devices() const { return devices_.shape().mesh_size(); }

bool MeshDeviceView::contains_device(chip_id_t device_id) const {
    return device_coordinates_.find(device_id) != device_coordinates_.end();
}

MeshCoordinate MeshDeviceView::find_device(chip_id_t device_id) const {
    auto it = device_coordinates_.find(device_id);
    TT_FATAL(it != device_coordinates_.end(), "Device not found in mesh: {}", device_id);
    return it->second;
}

chip_id_t MeshDeviceView::find_device_id(const MeshCoordinate& coord) const {
    TT_FATAL(contains(coord), "Coordinate {} not found in mesh {}", coord, devices_.shape());
    return devices_.at(coord)->id();
}

bool MeshDeviceView::is_mesh_2d() const { return shape_2d_.has_value(); }

std::vector<MeshCoordinate> MeshDeviceView::get_line_coordinates(
    size_t length, const Shape2D& mesh_shape, const Shape2D& mesh_offset) {
    // Iterate in a zigzag pattern from top-left to bottom-right, starting at the offset.
    std::vector<MeshCoordinate> line_coords;
    line_coords.reserve(length);
    const auto [num_rows, num_cols] = mesh_shape;
    auto [row_index, col_index] = mesh_offset;
    bool left_to_right = true;

    for (size_t i = 0; i < length && row_index < num_rows && col_index < num_cols; ++i) {
        line_coords.emplace_back(MeshCoordinate(row_index, col_index));

        if (left_to_right && col_index < num_cols - 1) {
            col_index++;
        } else if (!left_to_right && col_index > 0) {
            col_index--;
        } else {
            row_index++;
            left_to_right = !left_to_right;
        }
    }

    TT_FATAL(line_coords.size() == length, "Failed to get line coordinates");
    return line_coords;
}

std::vector<MeshCoordinate> MeshDeviceView::get_ring_coordinates(const Shape2D& ring_shape, const Shape2D& mesh_shape) {
    const auto [ring_rows, ring_cols] = ring_shape;
    TT_FATAL(ring_rows > 0 && ring_cols > 0, "Ring shape must not be empty along either dimension. Got {}", ring_shape);
    TT_FATAL(
        ring_rows <= mesh_shape.height() && ring_cols <= mesh_shape.width(),
        "Subgrid {} is out of mesh bounds {}",
        ring_shape,
        mesh_shape);

    const auto end_row = ring_rows - 1;
    const auto end_col = ring_cols - 1;

    std::vector<MeshCoordinate> boundary_coords;

    // Traverse the top row from left to right
    for (size_t col = 0; col <= end_col; ++col) {
        boundary_coords.emplace_back(MeshCoordinate{0, col});
    }

    // Traverse the rightmost column from top+1 to bottom
    for (size_t row = 1; row <= end_row; ++row) {
        boundary_coords.emplace_back(MeshCoordinate{row, end_col});
    }

    // Traverse the bottom row from right to left, if there is more than one row
    if (ring_rows > 1 and ring_cols > 1) {
        // Traverse the bottom row from right to left
        for (int col = static_cast<int>(end_col - 1); col >= 0; --col) {
            boundary_coords.emplace_back(MeshCoordinate{end_row, static_cast<size_t>(col)});
        }

        // Traverse the leftmost column from bottom-1 to top+1
        for (int row = static_cast<int>(end_row - 1); row > 0; --row) {
            boundary_coords.emplace_back(MeshCoordinate{static_cast<size_t>(row), 0});
        }
    }

    return boundary_coords;
}

std::vector<IDevice*> MeshDeviceView::get_line_devices() const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceView is not 2D!");
    auto boundary_coords =
        get_line_coordinates(devices_.shape().mesh_size(), *shape_2d_, /*mesh_offset=*/Shape2D(0, 0));
    return get_devices_from_coordinates(*this, boundary_coords);
}

std::vector<IDevice*> MeshDeviceView::get_ring_devices() const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceView is not 2D!");
    auto boundary_coords = get_ring_coordinates(*shape_2d_, *shape_2d_);
    return get_devices_from_coordinates(*this, boundary_coords);
}

MeshDeviceView::DeviceView MeshDeviceView::get_devices() const { return this->devices_.values(); }

}  // namespace tt::tt_metal::distributed
