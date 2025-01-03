// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <stdexcept>

#include "tt_metal/distributed/mesh_device_view.hpp"
#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/distributed/mesh_shape.hpp"

namespace tt::tt_metal::distributed {
namespace {

MeshDeviceView::DeviceView get_devices_from_coordinates(
    const MeshDeviceView& mesh, const std::vector<Coordinate>& coords) {
    MeshDeviceView::DeviceView devices;
    for (const auto& [row, col] : coords) {
        if (auto* device = mesh.get_device(row, col)) {
            devices.push_back(device);
        }
    }
    return devices;
}

}  // namespace

MeshDeviceView::MeshDeviceView(const MeshDevice& mesh) :
    top_left_(0, 0), bottom_right_(mesh.num_rows() - 1, mesh.num_cols() - 1) {
    for (size_t row = 0; row < mesh.num_rows(); ++row) {
        for (size_t col = 0; col < mesh.num_cols(); ++col) {
            if (auto* device = mesh.get_device(row, col)) {
                devices_.push_back(device);
                device_coordinates_[(device)->id()] = {row, col};
            }
        }
    }
}

MeshDeviceView::MeshDeviceView(const MeshDevice& mesh, Coordinate top_left, Coordinate bottom_right) :
    top_left_(0, 0), bottom_right_(Coordinate{bottom_right.row - top_left.row, bottom_right.col - top_left.col}) {
    for (size_t row = top_left.row; row <= bottom_right.row; ++row) {
        for (size_t col = top_left.col; col <= bottom_right.col; ++col) {
            if (auto* device = mesh.get_device(row, col)) {
                devices_.push_back(device);
                device_coordinates_[(device)->id()] = {row - top_left.row, col - top_left.col};
            }
        }
    }
    validate_coordinates();
}

MeshDeviceView::MeshDeviceView(std::vector<Device*> devices, const CoordinateMapper& mapper) :
    devices_(std::move(devices)) {
    initialize_from_devices(devices_, std::move(mapper));
}

Device* MeshDeviceView::get_device(size_t row, size_t col) const {
    for (const auto& device : devices_) {
        auto it = device_coordinates_.find(device->id());
        if (it != device_coordinates_.end() && it->second.row == row && it->second.col == col) {
            return device;
        }
    }
    return nullptr;
}

MeshDeviceView::DeviceView MeshDeviceView::get_devices(const Coordinate& start, const Coordinate& end) const {
    if (start.row > end.row || start.col > end.col) {
        log_fatal("Invalid coordinates: start {} must be less than or equal to end {}", start, end);
    }

    DeviceView devices_in_region;
    for (size_t row = start.row; row <= end.row; ++row) {
        for (size_t col = start.col; col <= end.col; ++col) {
            if (auto device = get_device(row, col)) {
                devices_in_region.push_back(device);
            }
        }
    }
    return devices_in_region;
}

MeshDeviceView::DeviceView MeshDeviceView::get_devices(const MeshShape& submesh_shape) const {
    return get_devices({0, 0}, {submesh_shape.num_rows - 1, submesh_shape.num_cols - 1});
}

MeshDeviceView::DeviceView MeshDeviceView::get_devices_on_row(size_t row) const {
    DeviceView row_devices;
    for (const auto& device : devices_) {
        auto it = device_coordinates_.find(device->id());
        if (it != device_coordinates_.end() && it->second.row == row) {
            row_devices.push_back(device);
        }
    }
    return row_devices;
}

MeshDeviceView::DeviceView MeshDeviceView::get_devices_on_column(size_t col) const {
    DeviceView col_devices;
    for (const auto& device : devices_) {
        auto it = device_coordinates_.find(device->id());
        if (it != device_coordinates_.end() && it->second.col == col) {
            col_devices.push_back(device);
        }
    }
    return col_devices;
}

MeshDeviceView::DeviceViews MeshDeviceView::get_row_views() const {
    MeshDeviceView::DeviceViews row_views;
    for (size_t row = top_left_.row; row <= bottom_right_.row; ++row) {
        row_views.push_back(get_devices_on_row(row));
    }
    return row_views;
}

MeshDeviceView::DeviceViews MeshDeviceView::get_column_views() const {
    MeshDeviceView::DeviceViews column_views;
    for (size_t col = top_left_.col; col <= bottom_right_.col; ++col) {
        column_views.push_back(get_devices_on_column(col));
    }
    return column_views;
}

bool MeshDeviceView::empty() const noexcept { return devices_.empty(); }

size_t MeshDeviceView::size() const noexcept { return devices_.size(); }

MeshShape MeshDeviceView::shape() const noexcept { return {num_rows(), num_cols()}; }

bool MeshDeviceView::contains(const Coordinate& coord) const noexcept {
    return coord.row >= top_left_.row && coord.row <= bottom_right_.row && coord.col >= top_left_.col &&
           coord.col <= bottom_right_.col;
}

const Device* MeshDeviceView::at(const Coordinate& coord) const noexcept {
    if (contains(coord)) {
        return get_device(coord.row, coord.col);
    }
    return nullptr;
}

bool MeshDeviceView::operator==(const MeshDeviceView& other) const {
    return devices_ == other.devices_ && device_coordinates_ == other.device_coordinates_ &&
           top_left_ == other.top_left_ && bottom_right_ == other.bottom_right_;
}

bool MeshDeviceView::contains_device(chip_id_t device_id) const {
    return device_coordinates_.find(device_id) != device_coordinates_.end();
}

Coordinate MeshDeviceView::find_device(chip_id_t device_id) const {
    auto it = device_coordinates_.find(device_id);
    if (it != device_coordinates_.end()) {
        return it->second;
    }
    TT_THROW("Device not found in mesh: {}", device_id);
}

chip_id_t MeshDeviceView::find_device_id(const Coordinate& coord) const {
    TT_FATAL(
        coord.row >= 0 and coord.row < num_rows() and coord.col >= 0 and coord.col < num_cols(),
        "Invalid coordinate: ({}, {})",
        coord.row,
        coord.col);
    return this->devices_.at(coord.row * num_cols() + coord.col)->id();
}

void MeshDeviceView::initialize_from_devices(const std::vector<Device*>& devices, const CoordinateMapper& mapper) {
    size_t min_row = std::numeric_limits<size_t>::max(), min_col = std::numeric_limits<size_t>::max();
    size_t max_row = std::numeric_limits<size_t>::min(), max_col = std::numeric_limits<size_t>::min();

    for (const auto& device : devices) {
        auto coord = mapper(device->id());
        if (!coord) {
            throw std::runtime_error("Failed to map device ID to coordinate");
        }

        device_coordinates_[device->id()] = *coord;
        min_row = std::min(min_row, coord->row);
        min_col = std::min(min_col, coord->col);
        max_row = std::max(max_row, coord->row);
        max_col = std::max(max_col, coord->col);
    }

    top_left_ = {min_row, min_col};
    bottom_right_ = {max_row, max_col};
}

std::vector<Coordinate> MeshDeviceView::get_line_coordinates(
    size_t length, const Coordinate& offset, size_t num_rows, size_t num_cols) {
    std::vector<Coordinate> line_coords;
    auto [row_index, col_index] = offset;
    bool left_to_right = true;

    for (size_t i = 0; i < length && row_index < num_rows && col_index < num_cols; ++i) {
        line_coords.emplace_back(Coordinate{row_index, col_index});

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

std::vector<Coordinate> MeshDeviceView::get_ring_coordinates(
    const MeshShape& ring_shape, const Coordinate& offset, size_t num_rows, size_t num_cols) const {
    auto [start_row, start_col] = offset;
    auto [ring_rows, ring_cols] = ring_shape;
    auto end_row = start_row + ring_rows - 1;
    auto end_col = start_col + ring_cols - 1;

    // Validate the specified subgrid
    std::vector<Coordinate> boundary_coords;
    if (start_row + ring_rows > num_rows || start_col + ring_cols > num_cols) {
        throw std::invalid_argument("Subgrid is out of mesh bounds.");
    }

    // Traverse the top row from left to right
    for (size_t col = start_col; col <= end_col; ++col) {
        boundary_coords.emplace_back(Coordinate{start_row, col});
    }

    // Traverse the rightmost column from top+1 to bottom
    for (size_t row = start_row + 1; row <= end_row; ++row) {
        boundary_coords.emplace_back(Coordinate{row, end_col});
    }

    // Traverse the bottom row from right to left, if there is more than one row
    if (ring_rows > 1 and ring_cols > 1) {
        // Traverse the bottom row from right to left
        for (int col = static_cast<int>(end_col - 1); col >= static_cast<int>(start_col); --col) {
            boundary_coords.emplace_back(Coordinate{end_row, static_cast<size_t>(col)});
        }

        // Traverse the leftmost column from bottom-1 to top+1
        for (int row = static_cast<int>(end_row - 1); row > static_cast<int>(start_row); --row) {
            boundary_coords.emplace_back(Coordinate{static_cast<size_t>(row), start_col});
        }
    }

    return boundary_coords;
}

void MeshDeviceView::validate_coordinates() const {
    if (top_left_.row > bottom_right_.row || top_left_.col > bottom_right_.col) {
        throw std::invalid_argument("Invalid coordinates: top_left must be less than or equal to bottom_right");
    }
}

MeshDeviceView::DeviceView MeshDeviceView::get_line_devices() const {
    auto boundary_coords =
        get_line_coordinates(this->num_rows() * this->num_cols(), this->top_left_, this->num_rows(), this->num_cols());
    return get_devices_from_coordinates(*this, boundary_coords);
}

MeshDeviceView::DeviceView MeshDeviceView::get_ring_devices() const {
    auto boundary_coords = get_ring_coordinates(shape(), this->top_left_, this->num_rows(), this->num_cols());
    return get_devices_from_coordinates(*this, boundary_coords);
}

MeshDeviceView::DeviceView MeshDeviceView::get_devices(MeshType type) const {
    switch (type) {
        case MeshType::RowMajor: return this->devices_;
        case MeshType::Ring: return this->get_ring_devices();
        case MeshType::Line: return this->get_line_devices();
        default: TT_THROW("Unsupported Mesh type: {}", type);
    }
}

}  // namespace tt::tt_metal::distributed
