// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/mesh_device_view.hpp"
#include "tt_metal/impl/device/mesh_device.hpp"
#include "tt_metal/impl/device/device_impl.hpp"
#include <algorithm>
#include <stdexcept>

namespace tt::tt_metal {

using MeshDevice = tt::tt_metal::MeshDevice;

MeshDeviceView::MeshDeviceView(const MeshDevice& mesh)
    : top_left_(0, 0), bottom_right_(mesh.num_rows() - 1, mesh.num_cols() - 1) {
    for (size_t row = 0; row < mesh.num_rows(); ++row) {
        for (size_t col = 0; col < mesh.num_cols(); ++col) {
            if (auto device = mesh.get_device(row, col)) {
                devices_.push_back(device);
                device_coordinates_[(device)->id()] = {row, col};
            }
        }
    }
}

MeshDeviceView::MeshDeviceView(const MeshDevice& mesh, Coordinate top_left, Coordinate bottom_right)
    : top_left_(top_left), bottom_right_(bottom_right) {
    for (size_t row = top_left.row; row <= bottom_right.row; ++row) {
        for (size_t col = top_left.col; col <= bottom_right.col; ++col) {
            if (auto device = mesh.get_device(row, col)) {
                devices_.push_back(device);
                device_coordinates_[(device)->id()] = {row, col};
            }
        }
    }
    validate_coordinates();
}

MeshDeviceView::MeshDeviceView(std::vector<device_pointer> devices, CoordinateMapper mapper)
    : devices_(std::move(devices)) {
    initialize_from_devices(devices_, std::move(mapper));
}

MeshDeviceView::device_pointer MeshDeviceView::get_device(size_t row, size_t col) {
    return const_cast<device_pointer>(std::as_const(*this).get_device(row, col));
}

MeshDeviceView::const_device_pointer MeshDeviceView::get_device(size_t row, size_t col) const {
    for (const auto& device : devices_) {
        auto it = device_coordinates_.find(device->id());
        if (it != device_coordinates_.end() && it->second.row == row && it->second.col == col) {
            return device;
        }
    }
    return nullptr;
}

const std::vector<MeshDeviceView::device_pointer>& MeshDeviceView::get_devices() const {
    return devices_;
}

MeshDeviceView::DeviceView MeshDeviceView::get_devices(const Coordinate& start, const Coordinate& end) {
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

MeshDeviceView::DeviceView MeshDeviceView::get_devices(const MeshShape& shape) {
    return get_devices({0, 0}, {shape.first - 1, shape.second - 1});
}

std::vector<MeshDeviceView::device_pointer> MeshDeviceView::get_devices_on_row(size_t row) const {
    std::vector<device_pointer> row_devices;
    for (const auto& device : devices_) {
        auto it = device_coordinates_.find(device->id());
        if (it != device_coordinates_.end() && it->second.row == row) {
            row_devices.push_back(device);
        }
    }
    return row_devices;
}

std::vector<MeshDeviceView::device_pointer> MeshDeviceView::get_devices_on_column(size_t col) const {
    std::vector<device_pointer> col_devices;
    for (const auto& device : devices_) {
        auto it = device_coordinates_.find(device->id());
        if (it != device_coordinates_.end() && it->second.col == col) {
            col_devices.push_back(device);
        }
    }
    return col_devices;
}

std::vector<std::vector<MeshDeviceView::device_pointer>> MeshDeviceView::get_row_views() const {
    std::vector<std::vector<device_pointer>> row_views;
    for (size_t row = top_left_.row; row <= bottom_right_.row; ++row) {
        row_views.push_back(get_devices_on_row(row));
    }
    return row_views;
}

std::vector<std::vector<MeshDeviceView::device_pointer>> MeshDeviceView::get_column_views() const {
    std::vector<std::vector<device_pointer>> column_views;
    for (size_t col = top_left_.col; col <= bottom_right_.col; ++col) {
        column_views.push_back(get_devices_on_column(col));
    }
    return column_views;
}

template<typename Pred>
MeshDeviceView MeshDeviceView::subview(Pred&& predicate) const {
    std::vector<device_pointer> filtered_devices;
    std::copy_if(devices_.begin(), devices_.end(), std::back_inserter(filtered_devices), std::forward<Pred>(predicate));
    return MeshDeviceView(filtered_devices, [this](int device_id) {
        auto it = device_coordinates_.find(device_id);
        return it != device_coordinates_.end() ? std::optional<Coordinate>(it->second) : std::nullopt;
    });
}

bool MeshDeviceView::empty() const noexcept {
    return devices_.empty();
}

size_t MeshDeviceView::size() const noexcept {
    return devices_.size();
}

std::pair<size_t, size_t> MeshDeviceView::shape() const noexcept {
    return {num_rows(), num_cols()};
}

bool MeshDeviceView::contains(const Coordinate& coord) const noexcept {
    return coord.row >= top_left_.row && coord.row <= bottom_right_.row &&
           coord.col >= top_left_.col && coord.col <= bottom_right_.col;
}

MeshDeviceView::const_device_pointer MeshDeviceView::at(const Coordinate& coord) const noexcept {
    if (contains(coord)) {
        return get_device(coord.row, coord.col);
    }
    return nullptr;
}

bool MeshDeviceView::operator==(const MeshDeviceView& other) const {
    return devices_ == other.devices_ &&
           device_coordinates_ == other.device_coordinates_ &&
           top_left_ == other.top_left_ &&
           bottom_right_ == other.bottom_right_;
}

Coordinate MeshDeviceView::find_device(chip_id_t device_id) const {
    auto it = device_coordinates_.find(device_id);
    if (it != device_coordinates_.end()) {
        return it->second;
    }
    TT_FATAL(false, fmt::format("Device not found in mesh: {}", device_id));
}

chip_id_t MeshDeviceView::find_device_id(const Coordinate& coord) const {
    TT_FATAL(coord.row >= 0 and coord.row < num_rows() and coord.col >= 0 and coord.col < num_cols(),
        fmt::format("Invalid coordinate: "));
    return this->devices_.at(coord.row * num_cols() + coord.col)->id();
}

void MeshDeviceView::initialize_from_devices(const std::vector<device_pointer>& devices, CoordinateMapper mapper) {
    std::size_t min_row = std::numeric_limits<std::size_t>::max(), min_col = std::numeric_limits<std::size_t>::max();
    std::size_t max_row = std::numeric_limits<std::size_t>::min(), max_col = std::numeric_limits<std::size_t>::min();

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

void MeshDeviceView::validate_coordinates() const {
    if (top_left_.row > bottom_right_.row || top_left_.col > bottom_right_.col) {
        throw std::invalid_argument("Invalid coordinates: top_left must be less than or equal to bottom_right");
    }
}

} // namespace tt::tt_metal
