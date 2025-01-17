// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <system_mesh.hpp>

#include "umd/device/types/cluster_descriptor_types.h"
#include "tt_metal/distributed/coordinate_translation.hpp"

namespace tt::tt_metal::distributed {

class SystemMesh::Impl {
private:
    std::unordered_map<MeshDeviceID, std::vector<chip_id_t>> assigned_devices_;
    std::unordered_map<MeshDeviceID, std::weak_ptr<MeshDevice>> assigned_mesh_device_devices_;

    MeshShape logical_mesh_shape_;
    CoordinateTranslationMap logical_to_physical_coordinates_;
    std::unordered_map<LogicalCoordinate, chip_id_t> logical_to_device_id_;
    std::unordered_map<PhysicalCoordinate, chip_id_t> physical_coordinate_to_device_id_;
    std::unordered_map<chip_id_t, PhysicalCoordinate> physical_device_id_to_coordinate_;

public:
    Impl() = default;
    ~Impl() = default;

    bool is_system_mesh_initialized() const;
    void initialize();
    const MeshShape& get_shape() const;
    size_t get_num_devices() const;
    std::vector<chip_id_t> get_mapped_physical_device_ids(const MeshDeviceConfig& config) const;
    std::vector<chip_id_t> request_available_devices(const MeshDeviceConfig& config) const;
    IDevice* get_device(const chip_id_t physical_device_id) const;
    void register_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device, const std::vector<IDevice*>& devices);

    chip_id_t get_physical_device_id(size_t logical_row_idx, size_t logical_col_idx) const;
};

// Implementation of public methods
bool SystemMesh::Impl::is_system_mesh_initialized() const { return !physical_coordinate_to_device_id_.empty(); }

void SystemMesh::Impl::initialize() {
    physical_device_id_to_coordinate_ = tt::Cluster::instance().get_user_chip_ethernet_coordinates();
    if (physical_device_id_to_coordinate_.empty()) {
        // Only WH has ethernet coordinates. Fabric will assign chip ids for BH
        auto arch = tt::Cluster::instance().arch();
        TT_FATAL(
            arch == ARCH::GRAYSKULL or arch == ARCH::BLACKHOLE,
            "Expected Wormhole chips to have ethernet coordinates assigned by cluster descriptor");
        const int num_detected_devices = tt::Cluster::instance().number_of_devices();
        for (auto chip_id = 0; chip_id < num_detected_devices; chip_id++) {
            PhysicalCoordinate coord{0, chip_id, 0, 0, 0};
            physical_device_id_to_coordinate_.emplace(chip_id, coord);
            physical_coordinate_to_device_id_.emplace(coord, chip_id);
        }
    } else {
        for (const auto& [chip_id, physical_coordinate] : this->physical_device_id_to_coordinate_) {
            physical_coordinate_to_device_id_.emplace(physical_coordinate, chip_id);
        }
    }

    std::tie(logical_to_physical_coordinates_, logical_mesh_shape_) = get_system_mesh_coordinate_translation_map();
    for (const auto& [logical_coordinate, physical_coordinate] : logical_to_physical_coordinates_) {
        auto physical_device_id_iter = physical_coordinate_to_device_id_.find(physical_coordinate);
        TT_FATAL(
            physical_device_id_iter != physical_coordinate_to_device_id_.end(),
            "Physical (Ethernet) coordinate: {} not found. Have you used `tt-topology` to flash the ethernet "
            "coordinates with the correct topology?",
            physical_coordinate);
        logical_to_device_id_.try_emplace(logical_coordinate, physical_device_id_iter->second);
    }
}

const MeshShape& SystemMesh::Impl::get_shape() const { return logical_mesh_shape_; }
size_t SystemMesh::Impl::get_num_devices() const {
    auto [num_rows, num_cols] = this->get_shape();
    return num_rows * num_cols;
}

chip_id_t SystemMesh::Impl::get_physical_device_id(size_t logical_row_idx, size_t logical_col_idx) const {
    TT_FATAL(
        logical_row_idx < logical_mesh_shape_.num_rows,
        "Row index out of bounds: {} >= {}",
        logical_row_idx,
        logical_mesh_shape_.num_rows);
    TT_FATAL(
        logical_col_idx < logical_mesh_shape_.num_cols,
        "Column index out of bounds: {} >= {}",
        logical_col_idx,
        logical_mesh_shape_.num_cols);
    auto logical_coordinate = Coordinate{logical_row_idx, logical_col_idx};
    return logical_to_device_id_.at(logical_coordinate);
}

std::vector<chip_id_t> SystemMesh::Impl::get_mapped_physical_device_ids(const MeshDeviceConfig& config) const {
    std::vector<chip_id_t> physical_device_ids;
    auto [system_mesh_rows, system_mesh_cols] = this->get_shape();
    auto [requested_num_rows, requested_num_cols] = config.mesh_shape;
    auto [row_offset, col_offset] = config.offset;

    // First check if total size fits
    TT_FATAL(
        requested_num_rows * requested_num_cols <= system_mesh_rows * system_mesh_cols,
        "Requested submesh is too big: {}x{}, SystemMesh shape: {}x{}",
        requested_num_rows,
        requested_num_cols,
        system_mesh_rows,
        system_mesh_cols);

    bool is_single_row_or_column = requested_num_rows == 1 or requested_num_cols == 1;
    if (is_single_row_or_column) {
        TT_FATAL(row_offset == 0 and col_offset == 0, "Row and column offsets unsupported for single row mesh");
        auto line_length = requested_num_rows * requested_num_cols;
        auto line_coords = MeshDeviceView::get_line_coordinates(
            line_length, Coordinate{row_offset, col_offset}, system_mesh_rows, system_mesh_cols);
        for (const auto& logical_coordinate : line_coords) {
            auto physical_device_id = logical_to_device_id_.at(logical_coordinate);
            physical_device_ids.push_back(physical_device_id);

            log_debug(
                LogMetal, "Logical coordinate: {}, Physical device ID: {}", logical_coordinate, physical_device_id);
        }
        return physical_device_ids;
    }
    bool requires_rotation = requested_num_rows > system_mesh_rows || requested_num_cols > system_mesh_cols;

    if (requires_rotation) {
        bool can_rotate = requested_num_rows <= system_mesh_cols && requested_num_cols <= system_mesh_rows;
        if (can_rotate) {
            // Rotate requested shape; row_offset and col_offset refer to original orientation
            std::swap(requested_num_rows, requested_num_cols);
        } else {
            TT_THROW(
                "User has requested a submesh that is too big and is not rotatable: {}x{} and SystemMesh is {}x{}.",
                requested_num_rows,
                requested_num_cols,
                system_mesh_rows,
                system_mesh_cols);
        }
    } else {
        // If no rotation, check dimensions directly
        TT_FATAL(
            requested_num_rows <= system_mesh_rows && requested_num_cols <= system_mesh_cols,
            "Requested submesh is too big: {}x{} and SystemMesh is {}x{}",
            requested_num_rows,
            requested_num_cols,
            system_mesh_rows,
            system_mesh_cols);
    }

    size_t original_rows = system_mesh_rows;
    size_t original_cols = system_mesh_cols;

    // Check that offsets fit in the original mesh
    TT_FATAL(
        row_offset + requested_num_rows <= original_rows,
        "Row offset + requested rows exceeds mesh size: {} + {} > {}",
        row_offset,
        requested_num_rows,
        original_rows);
    TT_FATAL(
        col_offset + requested_num_cols <= original_cols,
        "Column offset + requested columns exceeds mesh size: {} + {} > {}",
        col_offset,
        requested_num_cols,
        original_cols);

    // Map each submesh coordinate to the original logical coordinates
    for (size_t row = 0; row < requested_num_rows; row++) {
        for (size_t col = 0; col < requested_num_cols; col++) {
            Coordinate logical_coordinate;
            if (requires_rotation) {
                // After swapping requested_num_rows and requested_num_cols,
                // (row, col) now iterate over the rotated shape.
                size_t old_row = row_offset + row;  // top row
                size_t old_col = col_offset + col;  // increasing columns horizontally
                logical_coordinate = Coordinate{old_row, old_col};
            } else {
                logical_coordinate = Coordinate{row + row_offset, col + col_offset};
            }

            TT_FATAL(
                logical_coordinate.row < logical_mesh_shape_.num_rows,
                "Row coordinate out of bounds: {} >= {}",
                logical_coordinate.row,
                logical_mesh_shape_.num_rows);
            TT_FATAL(
                logical_coordinate.col < logical_mesh_shape_.num_cols,
                "Column coordinate out of bounds: {} >= {}",
                logical_coordinate.col,
                logical_mesh_shape_.num_cols);

            auto physical_device_id = logical_to_device_id_.at(logical_coordinate);
            physical_device_ids.push_back(physical_device_id);

            log_debug(
                LogMetal, "Logical coordinate: {}, Physical device ID: {}", logical_coordinate, physical_device_id);
        }
    }
    return physical_device_ids;
}

void SystemMesh::Impl::register_mesh_device(
    const std::shared_ptr<MeshDevice>& mesh_device, const std::vector<IDevice*>& devices) {
    std::vector<chip_id_t> physical_device_ids;
    for (auto device : devices) {
        physical_device_ids.push_back(device->id());
    }
    assigned_mesh_device_devices_.insert({mesh_device->id(), mesh_device});
    assigned_devices_.insert({mesh_device->id(), physical_device_ids});
}

std::vector<chip_id_t> SystemMesh::Impl::request_available_devices(const MeshDeviceConfig& config) const {
    auto [requested_num_rows, requested_num_cols] = config.mesh_shape;
    auto [max_num_rows, max_num_cols] = logical_mesh_shape_;
    auto [row_offset, col_offset] = config.offset;

    log_debug(
        LogMetal,
        "Mapping MeshDevice ({}x{}) with offset: {}, {}",
        requested_num_rows,
        requested_num_cols,
        row_offset,
        col_offset);

    return config.physical_device_ids.empty() ? this->get_mapped_physical_device_ids(config)
                                              : config.physical_device_ids;
}

SystemMesh::SystemMesh() : pimpl_(std::make_unique<Impl>()) {}
SystemMesh::~SystemMesh() = default;

SystemMesh& SystemMesh::instance() {
    static SystemMesh instance;
    if (!instance.pimpl_->is_system_mesh_initialized()) {
        instance.pimpl_->initialize();
    }
    return instance;
}

chip_id_t SystemMesh::get_physical_device_id(size_t logical_row_idx, size_t logical_col_idx) const {
    return pimpl_->get_physical_device_id(logical_row_idx, logical_col_idx);
}

const MeshShape& SystemMesh::get_shape() const { return pimpl_->get_shape(); }

size_t SystemMesh::get_num_devices() const { return pimpl_->get_num_devices(); }

void SystemMesh::register_mesh_device(
    const std::shared_ptr<MeshDevice>& mesh_device, const std::vector<IDevice*>& devices) {
    pimpl_->register_mesh_device(mesh_device, devices);
}

std::vector<chip_id_t> SystemMesh::request_available_devices(const MeshDeviceConfig& config) const {
    return pimpl_->request_available_devices(config);
}

std::vector<chip_id_t> SystemMesh::get_mapped_physical_device_ids(const MeshDeviceConfig& config) const {
    return pimpl_->get_mapped_physical_device_ids(config);
}

}  // namespace tt::tt_metal::distributed
