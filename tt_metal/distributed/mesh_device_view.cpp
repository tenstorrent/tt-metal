// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_device.hpp>
#include <mesh_device_view.hpp>
#include "mesh_device_view_impl.hpp"
#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <vector>
#include <algorithm>

#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>
#include <climits>
#include "device.hpp"
#include "mesh_config.hpp"
#include "mesh_coord.hpp"
#include "shape2d.hpp"
#include "shape_base.hpp"
#include <tt-metalium/maybe_remote.hpp>

namespace tt::tt_metal::distributed {
namespace {

std::vector<IDevice*> get_devices_from_coordinates(
    const MeshDeviceViewImpl& mesh, const std::vector<MeshCoordinate>& coords) {
    std::vector<IDevice*> devices;
    for (const auto& coord : coords) {
        if (auto* device = mesh.get_device(coord)) {
            devices.push_back(device);
        }
    }
    return devices;
}

std::vector<tt::tt_fabric::FabricNodeId> get_fabric_node_ids_from_coordinates(
    const MeshDeviceViewImpl& mesh, const std::vector<MeshCoordinate>& coords) {
    std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids;
    fabric_node_ids.reserve(coords.size());
    for (const auto& coord : coords) {
        fabric_node_ids.push_back(mesh.get_fabric_node_id(coord));
    }
    return fabric_node_ids;
}

}  // namespace

// MeshDeviceViewImpl implementations

MeshDeviceViewImpl::MeshDeviceViewImpl(
    const MeshShape& shape,
    const std::vector<IDevice*>& devices,
    const std::vector<tt::tt_fabric::FabricNodeId>& fabric_node_ids) :
    MeshDeviceViewImpl(shape, wrap_to_maybe_remote(devices), fabric_node_ids) {}

MeshDeviceViewImpl::MeshDeviceViewImpl(
    const MeshShape& shape,
    const std::vector<MaybeRemote<IDevice*>>& devices,
    const std::vector<tt::tt_fabric::FabricNodeId>& fabric_node_ids) :
    devices_(shape, devices), fabric_node_ids_(shape, fabric_node_ids), mesh_id_(fabric_node_ids.front().mesh_id) {
    if (devices_.shape().dims() == 2) {
        shape_2d_ = Shape2D(devices_.shape()[0], devices_.shape()[1]);
    }

    TT_FATAL(
        std::all_of(
            fabric_node_ids.begin(),
            fabric_node_ids.end(),
            [this](const auto& fabric_node_id) { return fabric_node_id.mesh_id == mesh_id_; }),
        "All fabric node ids in MeshDeviceViewImpl must have the same mesh id: {}",
        *mesh_id_);

    // Build coordinate map.
    for (const auto& [coord, maybe_device] : devices_) {
        maybe_device.if_local([this, &coord](const auto& device) { device_coordinates_.emplace(device->id(), coord); });
    }
}

std::vector<IDevice*> MeshDeviceViewImpl::get_devices(const MeshCoordinateRange& range) const {
    std::vector<IDevice*> devices_in_region;
    for (const auto& coord : range) {
        devices_.at(coord).if_local([&devices_in_region](const auto& device) { devices_in_region.push_back(device); });
    }
    return devices_in_region;
}

std::vector<IDevice*> MeshDeviceViewImpl::get_devices() const { return extract_locals(devices_.values()); }

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceViewImpl::get_fabric_node_ids(
    const MeshCoordinateRange& range) const {
    std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids_in_region;
    fabric_node_ids_in_region.reserve(range.shape().mesh_size());
    for (const auto& coord : range) {
        fabric_node_ids_in_region.push_back(get_fabric_node_id(coord));
    }
    return fabric_node_ids_in_region;
}

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceViewImpl::get_fabric_node_ids() const {
    return fabric_node_ids_.values();
}

std::vector<IDevice*> MeshDeviceViewImpl::get_devices_on_row(size_t row) const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceViewImpl is not 2D!");
    TT_FATAL(row < shape_2d_->height(), "Row index out of bounds: {}", row);
    std::vector<IDevice*> row_devices;
    row_devices.reserve(shape_2d_->width());
    for (int col = 0; col < shape_2d_->width(); ++col) {
        const auto& coord = MeshCoordinate(row, col);
        devices_.at(coord).if_local([&row_devices](const auto& device) { row_devices.push_back(device); });
    }
    return row_devices;
}

std::vector<IDevice*> MeshDeviceViewImpl::get_devices_on_column(size_t col) const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceViewImpl is not 2D!");
    TT_FATAL(col < shape_2d_->width(), "Column index out of bounds: {}", col);
    std::vector<IDevice*> col_devices;
    col_devices.reserve(shape_2d_->height());
    for (int row = 0; row < shape_2d_->height(); ++row) {
        const auto& coord = MeshCoordinate(row, col);
        devices_.at(coord).if_local([&col_devices](const auto& device) { col_devices.push_back(device); });
    }
    return col_devices;
}

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceViewImpl::get_fabric_node_ids_on_row(size_t row) const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceViewImpl is not 2D!");
    TT_FATAL(row < shape_2d_->height(), "Row index out of bounds: {}", row);
    std::vector<tt::tt_fabric::FabricNodeId> row_fabric_node_ids;
    row_fabric_node_ids.reserve(shape_2d_->width());
    for (int col = 0; col < shape_2d_->width(); ++col) {
        row_fabric_node_ids.push_back(get_fabric_node_id(MeshCoordinate(row, col)));
    }
    return row_fabric_node_ids;
}

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceViewImpl::get_fabric_node_ids_on_column(size_t col) const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceViewImpl is not 2D!");
    TT_FATAL(col < shape_2d_->width(), "Column index out of bounds: {}", col);
    std::vector<tt::tt_fabric::FabricNodeId> col_fabric_node_ids;
    col_fabric_node_ids.reserve(shape_2d_->height());
    for (int row = 0; row < shape_2d_->height(); ++row) {
        col_fabric_node_ids.push_back(get_fabric_node_id(MeshCoordinate(row, col)));
    }
    return col_fabric_node_ids;
}

bool MeshDeviceViewImpl::empty() const noexcept { return devices_.shape().mesh_size() == 0; }
size_t MeshDeviceViewImpl::size() const noexcept { return devices_.shape().mesh_size(); }
const MeshShape& MeshDeviceViewImpl::shape() const noexcept { return devices_.shape(); }
tt::tt_fabric::MeshId MeshDeviceViewImpl::mesh_id() const noexcept { return mesh_id_; }

bool MeshDeviceViewImpl::contains(const MeshCoordinate& coord) const noexcept {
    return devices_.coord_range().contains(coord);
}

IDevice* MeshDeviceViewImpl::get_device(const MeshCoordinate& coord) const {
    if (!contains(coord)) {
        return nullptr;
    }
    const auto& maybe_device = devices_.at(coord);
    TT_FATAL(maybe_device.is_local(), "Cannot get device for remote device at coordinate {}", coord);
    return *maybe_device;
}

tt::tt_fabric::FabricNodeId MeshDeviceViewImpl::get_fabric_node_id(const MeshCoordinate& coord) const {
    TT_FATAL(contains(coord), "Coordinate {} not found in mesh {}", coord, devices_.shape());
    return fabric_node_ids_.at(coord);
}

size_t MeshDeviceViewImpl::num_rows() const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceViewImpl is not 2D!");
    return shape_2d_->height();
}
size_t MeshDeviceViewImpl::num_cols() const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceViewImpl is not 2D!");
    return shape_2d_->width();
}
size_t MeshDeviceViewImpl::num_devices() const { return devices_.shape().mesh_size(); }

MeshCoordinate MeshDeviceViewImpl::find_device(ChipId device_id) const {
    auto it = device_coordinates_.find(device_id);
    TT_FATAL(it != device_coordinates_.end(), "Device not found in mesh: {}", device_id);
    return it->second;
}

bool MeshDeviceViewImpl::is_mesh_2d() const { return shape_2d_.has_value(); }

std::vector<MeshCoordinate> MeshDeviceViewImpl::get_line_coordinates(
    size_t length, const Shape2D& mesh_shape, const Shape2D& mesh_offset) {
    const auto [num_rows, num_cols] = mesh_shape;
    auto [start_row, start_col] = mesh_offset;

    // Validate starting position
    TT_FATAL(
        start_row < num_rows && start_col < num_cols,
        "Mesh offset ({}, {}) is out of bounds for mesh shape ({}, {})",
        start_row,
        start_col,
        num_rows,
        num_cols);

    // Iterate in a zigzag pattern from top-left to bottom-right, starting at the offset.
    std::vector<MeshCoordinate> line_coords;
    line_coords.reserve(length);

    // NOTE: Special case: For 2x4 or 4x2 mesh shapes, use perimeter traversal to avoid snake patterns
    // that cause fabric initialization issues on T3K
    // https://github.com/tenstorrent/tt-metal/issues/33737
    if (mesh_shape == Shape2D(2, 4) || mesh_shape == Shape2D(4, 2)) {
        auto ring_coords = get_ring_coordinates(mesh_shape, mesh_shape);
        MeshCoordinate start_coord(start_row, start_col);
        auto start_it = std::find(ring_coords.begin(), ring_coords.end(), start_coord);
        TT_FATAL(
            start_it != ring_coords.end(), "Mesh offset ({}, {}) not found in ring coordinates", start_row, start_col);

        // check the length is less than or equal to the number of ring coordinates
        TT_FATAL(
            length <= ring_coords.size(),
            "Length {} is greater than the number of ring coordinates {}",
            length,
            ring_coords.size());

        size_t start_idx = std::distance(ring_coords.begin(), start_it);
        for (size_t i = 0; i < length; ++i) {
            line_coords.push_back(ring_coords[(start_idx + i) % ring_coords.size()]);
        }
        return line_coords;
    }

    const MeshCoordinate start_coord(start_row, start_col);
    // Lambda to check if two coordinates are adjacent (direct neighbors only: up, down, left, right)
    // Does NOT consider diagonal neighbors
    auto are_adjacent = [](const MeshCoordinate& a, const MeshCoordinate& b) -> bool {
        const size_t row_diff = (a[0] > b[0]) ? (a[0] - b[0]) : (b[0] - a[0]);
        const size_t col_diff = (a[1] > b[1]) ? (a[1] - b[1]) : (b[1] - a[1]);
        // Adjacent means exactly one dimension differs by 1 and the other by 0
        // This excludes diagonal neighbors (where both differ by 1)
        return (row_diff == 1 && col_diff == 0) || (row_diff == 0 && col_diff == 1);
    };

    // Lambda to get valid neighbors (not checking visited - that's done in DFS)
    auto get_neighbors = [&](const MeshCoordinate& coord) -> std::vector<MeshCoordinate> {
        std::vector<MeshCoordinate> neighbors;
        const size_t row = coord[0];
        const size_t col = coord[1];

        if (col + 1 < num_cols) {
            neighbors.emplace_back(row, col + 1);
        }
        if (row + 1 < num_rows) {
            neighbors.emplace_back(row + 1, col);
        }
        if (col > 0) {
            neighbors.emplace_back(row, col - 1);
        }
        if (row > 0) {
            neighbors.emplace_back(row - 1, col);
        }

        return neighbors;
    };

    // Recursive DFS helper with backtracking
    // First tries to find a ring path, then falls back to any valid path if ring is impossible
    std::function<bool(std::vector<MeshCoordinate>&, std::unordered_set<MeshCoordinate>&, bool require_ring)> dfs =
        [&](std::vector<MeshCoordinate>& path, std::unordered_set<MeshCoordinate>& visited, bool require_ring) -> bool {
        if (path.size() >= length) {
            return true;
        }

        MeshCoordinate current = path.back();

        // Get unvisited neighbors
        auto neighbors = get_neighbors(current);
        for (const auto& neighbor : neighbors) {
            if (visited.contains(neighbor)) {
                continue;
            }

            // If this is the last coordinate and we require a ring, it must be adjacent to start
            if (require_ring && path.size() == length - 1) {
                if (!are_adjacent(neighbor, start_coord)) {
                    continue;
                }
            }

            // Try this neighbor
            path.push_back(neighbor);
            visited.insert(neighbor);

            if (dfs(path, visited, require_ring)) {
                return true;
            }

            // Backtrack
            path.pop_back();
            visited.erase(neighbor);
        }

        return false;
    };

    // Initialize DFS
    std::unordered_set<MeshCoordinate> visited;
    visited.insert(start_coord);
    line_coords.push_back(start_coord);

    // First try to find a ring path (preferred)
    bool found_path = dfs(line_coords, visited, /*require_ring=*/true);

    // If ring not possible, fall back to any valid path
    if (!found_path) {
        // Reset and try without ring requirement
        line_coords.clear();
        visited.clear();
        visited.insert(start_coord);
        line_coords.push_back(start_coord);
        found_path = dfs(line_coords, visited, /*require_ring=*/false);
    }

    TT_FATAL(
        found_path && line_coords.size() == length,
        "Failed to get line coordinates, got {} coordinates, expected {}",
        line_coords.size(),
        length);
    return line_coords;
}

std::vector<MeshCoordinate> MeshDeviceViewImpl::get_ring_coordinates(
    const Shape2D& ring_shape, const Shape2D& mesh_shape) {
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
        boundary_coords.emplace_back(MeshCoordinate{0, static_cast<uint32_t>(col)});
    }

    // Traverse the rightmost column from top+1 to bottom
    for (size_t row = 1; row <= end_row; ++row) {
        boundary_coords.emplace_back(MeshCoordinate{static_cast<uint32_t>(row), static_cast<uint32_t>(end_col)});
    }

    // Traverse the bottom row from right to left, if there is more than one row
    if (ring_rows > 1 and ring_cols > 1) {
        // Traverse the bottom row from right to left
        for (int col = static_cast<int>(end_col - 1); col >= 0; --col) {
            boundary_coords.emplace_back(MeshCoordinate{static_cast<uint32_t>(end_row), static_cast<uint32_t>(col)});
        }

        // Traverse the leftmost column from bottom-1 to top+1
        for (int row = static_cast<int>(end_row - 1); row > 0; --row) {
            boundary_coords.emplace_back(MeshCoordinate{static_cast<uint32_t>(row), 0});
        }
    }

    return boundary_coords;
}

std::vector<MeshCoordinate> MeshDeviceViewImpl::get_line_coordinates() const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceViewImpl is not 2D!");
    return get_line_coordinates(devices_.shape().mesh_size(), *shape_2d_, /*mesh_offset=*/Shape2D(0, 0));
}

std::vector<MeshCoordinate> MeshDeviceViewImpl::get_ring_coordinates() const {
    TT_FATAL(shape_2d_.has_value(), "MeshDeviceViewImpl is not 2D!");
    return get_ring_coordinates(*shape_2d_, *shape_2d_);
}

std::vector<IDevice*> MeshDeviceViewImpl::get_line_devices() const {
    return get_devices_from_coordinates(*this, get_line_coordinates());
}

std::vector<IDevice*> MeshDeviceViewImpl::get_ring_devices() const {
    return get_devices_from_coordinates(*this, get_ring_coordinates());
}

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceViewImpl::get_line_fabric_node_ids() const {
    return get_fabric_node_ids_from_coordinates(*this, get_line_coordinates());
}

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceViewImpl::get_ring_fabric_node_ids() const {
    return get_fabric_node_ids_from_coordinates(*this, get_ring_coordinates());
}

bool MeshDeviceViewImpl::is_local(const MeshCoordinate& coord) const {
    TT_FATAL(contains(coord), "Coordinate {} not found in mesh {}", coord, devices_.shape());
    return devices_.at(coord).is_local();
}

MeshCoordinateRange MeshDeviceViewImpl::get_local_mesh_coord_range() const {
    const MeshShape& mesh_shape = shape();
    const size_t num_dims = mesh_shape.dims();

    // Initialize min and max coordinates
    tt::stl::SmallVector<uint32_t> min_coords(num_dims, UINT32_MAX);
    tt::stl::SmallVector<uint32_t> max_coords(num_dims, 0);
    bool found_local = false;

    // Iterate through all coordinates in the mesh
    for (const auto& coord : MeshCoordinateRange(mesh_shape)) {
        if (is_local(coord)) {
            found_local = true;
            for (size_t dim = 0; dim < num_dims; ++dim) {
                min_coords[dim] = std::min(min_coords[dim], coord[dim]);
                max_coords[dim] = std::max(max_coords[dim], coord[dim]);
            }
        }
    }

    TT_FATAL(found_local, "No local devices found in mesh device");

    MeshCoordinate start_coord(tt::stl::Span<const uint32_t>(min_coords.data(), num_dims));
    MeshCoordinate end_coord(tt::stl::Span<const uint32_t>(max_coords.data(), num_dims));

    return MeshCoordinateRange(start_coord, end_coord);
}

// MeshDeviceView implementations

MeshDeviceView::MeshDeviceView(
    const MeshShape& shape,
    const std::vector<IDevice*>& devices,
    const std::vector<tt::tt_fabric::FabricNodeId>& fabric_node_ids) :
    pimpl_(std::make_unique<MeshDeviceViewImpl>(shape, devices, fabric_node_ids)) {}

MeshDeviceView::MeshDeviceView(
    const MeshShape& shape,
    const std::vector<MaybeRemote<IDevice*>>& devices,
    const std::vector<tt::tt_fabric::FabricNodeId>& fabric_node_ids) :
    pimpl_(std::make_unique<MeshDeviceViewImpl>(shape, devices, fabric_node_ids)) {}

MeshDeviceView::~MeshDeviceView() = default;

MeshDeviceView::MeshDeviceView(const MeshDeviceView& other) :
    pimpl_(std::make_unique<MeshDeviceViewImpl>(*other.pimpl_)) {}

MeshDeviceView& MeshDeviceView::operator=(const MeshDeviceView& other) {
    if (this != &other) {
        pimpl_ = std::make_unique<MeshDeviceViewImpl>(*other.pimpl_);
    }
    return *this;
}

MeshDeviceView::MeshDeviceView(MeshDeviceView&&) noexcept = default;
MeshDeviceView& MeshDeviceView::operator=(MeshDeviceView&&) noexcept = default;

std::vector<IDevice*> MeshDeviceView::get_devices(const MeshCoordinateRange& range) const {
    return pimpl_->get_devices(range);
}

std::vector<IDevice*> MeshDeviceView::get_devices() const { return pimpl_->get_devices(); }

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceView::get_fabric_node_ids(const MeshCoordinateRange& range) const {
    return pimpl_->get_fabric_node_ids(range);
}

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceView::get_fabric_node_ids() const {
    return pimpl_->get_fabric_node_ids();
}

std::vector<IDevice*> MeshDeviceView::get_devices_on_row(size_t row) const { return pimpl_->get_devices_on_row(row); }

std::vector<IDevice*> MeshDeviceView::get_devices_on_column(size_t col) const {
    return pimpl_->get_devices_on_column(col);
}

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceView::get_fabric_node_ids_on_row(size_t row) const {
    return pimpl_->get_fabric_node_ids_on_row(row);
}

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceView::get_fabric_node_ids_on_column(size_t col) const {
    return pimpl_->get_fabric_node_ids_on_column(col);
}

bool MeshDeviceView::empty() const noexcept { return pimpl_->empty(); }

size_t MeshDeviceView::size() const noexcept { return pimpl_->size(); }

const MeshShape& MeshDeviceView::shape() const noexcept { return pimpl_->shape(); }

tt::tt_fabric::MeshId MeshDeviceView::mesh_id() const noexcept { return pimpl_->mesh_id(); }

bool MeshDeviceView::contains(const MeshCoordinate& coord) const noexcept { return pimpl_->contains(coord); }

IDevice* MeshDeviceView::get_device(const MeshCoordinate& coord) const { return pimpl_->get_device(coord); }

tt::tt_fabric::FabricNodeId MeshDeviceView::get_fabric_node_id(const MeshCoordinate& coord) const {
    return pimpl_->get_fabric_node_id(coord);
}

size_t MeshDeviceView::num_rows() const { return pimpl_->num_rows(); }

size_t MeshDeviceView::num_cols() const { return pimpl_->num_cols(); }

size_t MeshDeviceView::num_devices() const { return pimpl_->num_devices(); }

MeshCoordinate MeshDeviceView::find_device(ChipId device_id) const { return pimpl_->find_device(device_id); }

bool MeshDeviceView::is_mesh_2d() const { return pimpl_->is_mesh_2d(); }

std::vector<MeshCoordinate> MeshDeviceView::get_line_coordinates(
    size_t length, const Shape2D& mesh_shape, const Shape2D& mesh_offset) {
    return MeshDeviceViewImpl::get_line_coordinates(length, mesh_shape, mesh_offset);
}

std::vector<MeshCoordinate> MeshDeviceView::get_line_coordinates() const { return pimpl_->get_line_coordinates(); }

std::vector<MeshCoordinate> MeshDeviceView::get_ring_coordinates(const Shape2D& ring_shape, const Shape2D& mesh_shape) {
    return MeshDeviceViewImpl::get_ring_coordinates(ring_shape, mesh_shape);
}

std::vector<MeshCoordinate> MeshDeviceView::get_ring_coordinates() const { return pimpl_->get_ring_coordinates(); }

std::vector<IDevice*> MeshDeviceView::get_line_devices() const { return pimpl_->get_line_devices(); }

std::vector<IDevice*> MeshDeviceView::get_ring_devices() const { return pimpl_->get_ring_devices(); }

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceView::get_line_fabric_node_ids() const {
    return pimpl_->get_line_fabric_node_ids();
}

std::vector<tt::tt_fabric::FabricNodeId> MeshDeviceView::get_ring_fabric_node_ids() const {
    return pimpl_->get_ring_fabric_node_ids();
}

bool MeshDeviceView::is_local(const MeshCoordinate& coord) const { return pimpl_->is_local(coord); }

MeshCoordinateRange MeshDeviceView::get_local_mesh_coord_range() const { return pimpl_->get_local_mesh_coord_range(); }

std::vector<MaybeRemote<IDevice*>>::const_iterator MeshDeviceView::begin() const { return pimpl_->begin(); }

std::vector<MaybeRemote<IDevice*>>::const_iterator MeshDeviceView::end() const { return pimpl_->end(); }

}  // namespace tt::tt_metal::distributed
