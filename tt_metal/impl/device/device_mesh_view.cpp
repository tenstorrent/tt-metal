// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device_mesh_view.hpp"
#include "tt_metal/impl/device/device_mesh.hpp"
#include <algorithm>
#include <stdexcept>

// Utils to traverse devices in a ring to support AllGather operations
bool ham_cycle_dfs(
    const std::vector<std::vector<int>>& adj, std::vector<int>& path, std::vector<int>& visited, int pos) {
    if (pos == path.size()) {
        if (adj[path[pos - 1]][path[0]] == 1)
            return true;
        else
            return false;
    }

    for (int v = 0; v < adj.size(); v++) {
        if (adj[path[pos - 1]][v] == 1 and visited[v] == 0) {
            path[pos] = v;
            visited[v] = 1;

            if (ham_cycle_dfs(adj, path, visited, pos + 1) == true)
                return true;

            path[pos] = -1;
            visited[v] = 0;
        }
    }

    return false;
}

std::vector<int> get_hamiltonian_cycle(std::vector<std::vector<int>>& adj, int s, int N) {
    std::vector<int> path(N, -1);
    std::vector<int> visited(adj.size(), 0);

    path[0] = s;
    visited[s] = 1;
    if (ham_cycle_dfs(adj, path, visited, 1) == false) {
        TT_FATAL(false, fmt::format("Could not find ring of devices starting from chip {} with size.", s, N));
        return {};
    }
    return path;
}

namespace tt::tt_metal {

using DeviceMesh = tt::tt_metal::DeviceMesh;

DeviceMeshView::DeviceMeshView(const DeviceMesh& mesh)
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

DeviceMeshView::DeviceMeshView(const DeviceMesh& mesh, Coordinate top_left, Coordinate bottom_right)
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

DeviceMeshView::DeviceMeshView(std::vector<device_pointer> devices, CoordinateMapper mapper)
    : devices_(std::move(devices)) {
    initialize_from_devices(devices_, std::move(mapper));
}

DeviceMeshView::device_pointer DeviceMeshView::get_device(size_t row, size_t col) {
    return const_cast<device_pointer>(std::as_const(*this).get_device(row, col));
}

DeviceMeshView::const_device_pointer DeviceMeshView::get_device(size_t row, size_t col) const {
    for (const auto& device : devices_) {
        auto it = device_coordinates_.find(device->id());
        if (it != device_coordinates_.end() && it->second.row == row && it->second.col == col) {
            return device;
        }
    }
    return nullptr;
}

const std::vector<DeviceMeshView::device_pointer>& DeviceMeshView::get_devices() const {
    return devices_;
}

// Mesh Traveral APIs
DeviceMeshView::DeviceView DeviceMeshView::get_devices(const Coordinate& start, const Coordinate& end) {
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

DeviceMeshView::DeviceView DeviceMeshView::get_devices(const DeviceGrid& shape) {
    return get_devices({0, 0}, {shape.first - 1, shape.second - 1});
}

std::vector<DeviceMeshView::device_pointer> DeviceMeshView::get_devices_on_row(size_t row) const {
    std::vector<device_pointer> row_devices;
    for (const auto& device : devices_) {
        auto it = device_coordinates_.find(device->id());
        if (it != device_coordinates_.end() && it->second.row == row) {
            row_devices.push_back(device);
        }
    }
    return row_devices;
}

std::vector<DeviceMeshView::device_pointer> DeviceMeshView::get_devices_on_column(size_t col) const {
    std::vector<device_pointer> col_devices;
    for (const auto& device : devices_) {
        auto it = device_coordinates_.find(device->id());
        if (it != device_coordinates_.end() && it->second.col == col) {
            col_devices.push_back(device);
        }
    }
    return col_devices;
}

std::vector<std::vector<DeviceMeshView::device_pointer>> DeviceMeshView::get_row_views() const {
    std::vector<std::vector<device_pointer>> row_views;
    for (size_t row = top_left_.row; row <= bottom_right_.row; ++row) {
        row_views.push_back(get_devices_on_row(row));
    }
    return row_views;
}

std::vector<std::vector<DeviceMeshView::device_pointer>> DeviceMeshView::get_column_views() const {
    std::vector<std::vector<device_pointer>> column_views;
    for (size_t col = top_left_.col; col <= bottom_right_.col; ++col) {
        column_views.push_back(get_devices_on_column(col));
    }
    return column_views;
}

// Ring Traversal APIs
DeviceMeshView::DeviceView DeviceMeshView::get_devices_on_ring(
    std::vector<DeviceMeshView::device_pointer> devices, int start_device_id, int num_devices_in_ring) const {
    // Returns the indices of the devices in the ring starting from the start_device_id
    // i.e. there are connections from devices[device_ring_idx[i]] to devices[device_ring_idx[i+1]] for all i, with wrap
    std::vector<std::vector<int>> adj(devices.size(), std::vector<int>(devices.size(), 0));
    int start_index = -1;
    for (std::uint32_t i = 0; i < devices.size(); ++i) {
        const auto& device = devices[i];
        if (device->id() == start_device_id) {
            start_index = i;
        }
        for (const auto& connected_device_id : device->get_ethernet_connected_device_ids()) {
            for (std::uint32_t j = 0; j < devices.size(); ++j) {
                if (devices[j]->id() == connected_device_id) {
                    adj[i][j] = 1;
                }
            }
        }
    }
    TT_FATAL(start_index != -1, "Could not find start device in the list of devices");

    const auto& device_ring_idx = get_hamiltonian_cycle(adj, start_device_id, num_devices_in_ring);

    std::vector<DeviceMeshView::device_pointer> device_ring;
    device_ring.reserve(device_ring_idx.size());

    for (const auto& idx : device_ring_idx) {
        device_ring.push_back(devices[idx]);
    }

    return device_ring;
}

template<typename Pred>
DeviceMeshView DeviceMeshView::subview(Pred&& predicate) const {
    std::vector<device_pointer> filtered_devices;
    std::copy_if(devices_.begin(), devices_.end(), std::back_inserter(filtered_devices), std::forward<Pred>(predicate));
    return DeviceMeshView(filtered_devices, [this](int device_id) {
        auto it = device_coordinates_.find(device_id);
        return it != device_coordinates_.end() ? std::optional<Coordinate>(it->second) : std::nullopt;
    });
}

bool DeviceMeshView::empty() const noexcept {
    return devices_.empty();
}

size_t DeviceMeshView::size() const noexcept {
    return devices_.size();
}

std::pair<size_t, size_t> DeviceMeshView::shape() const noexcept {
    return {num_rows(), num_cols()};
}

bool DeviceMeshView::contains(const Coordinate& coord) const noexcept {
    return coord.row >= top_left_.row && coord.row <= bottom_right_.row &&
           coord.col >= top_left_.col && coord.col <= bottom_right_.col;
}

DeviceMeshView::const_device_pointer DeviceMeshView::at(const Coordinate& coord) const noexcept {
    if (contains(coord)) {
        return get_device(coord.row, coord.col);
    }
    return nullptr;
}

bool DeviceMeshView::operator==(const DeviceMeshView& other) const {
    return devices_ == other.devices_ &&
           device_coordinates_ == other.device_coordinates_ &&
           top_left_ == other.top_left_ &&
           bottom_right_ == other.bottom_right_;
}

Coordinate DeviceMeshView::find_device(chip_id_t device_id) const {
    auto it = device_coordinates_.find(device_id);
    if (it != device_coordinates_.end()) {
        return it->second;
    }
    TT_FATAL(false, fmt::format("Device not found in mesh: {}", device_id));
}

chip_id_t DeviceMeshView::find_device_id(const Coordinate& coord) const {
    TT_FATAL(coord.row >= 0 and coord.row < num_rows() and coord.col >= 0 and coord.col < num_cols(),
        fmt::format("Invalid coordinate: "));
    return this->devices_.at(coord.row * num_cols() + coord.col)->id();
}

void DeviceMeshView::initialize_from_devices(const std::vector<device_pointer>& devices, CoordinateMapper mapper) {
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

void DeviceMeshView::validate_coordinates() const {
    if (top_left_.row > bottom_right_.row || top_left_.col > bottom_right_.col) {
        throw std::invalid_argument("Invalid coordinates: top_left must be less than or equal to bottom_right");
    }
}

} // namespace tt::tt_metal
