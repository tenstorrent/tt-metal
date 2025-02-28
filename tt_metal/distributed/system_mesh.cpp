// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <system_mesh.hpp>

#include "small_vector.hpp"
#include "umd/device/types/cluster_descriptor_types.h"
#include "tt_metal/distributed/coordinate_translation.hpp"

#include "indestructible.hpp"
#include "mesh_coord.hpp"
#include "tt_cluster.hpp"

namespace tt::tt_metal::distributed {

class SystemMesh::Impl {
private:
    MeshShape logical_mesh_shape_;
    CoordinateTranslationMap logical_to_physical_coordinates_;
    std::unordered_map<MeshCoordinate, chip_id_t> logical_to_device_id_;
    std::unordered_map<PhysicalCoordinate, chip_id_t> physical_coordinate_to_device_id_;
    std::unordered_map<chip_id_t, PhysicalCoordinate> physical_device_id_to_coordinate_;

public:
    Impl() = default;

    bool is_system_mesh_initialized() const;
    void initialize();
    const MeshShape& get_shape() const;
    std::vector<chip_id_t> get_mapped_physical_device_ids(const MeshDeviceConfig& config) const;
    std::vector<chip_id_t> request_available_devices(const MeshDeviceConfig& config) const;
    chip_id_t get_physical_device_id(const MeshCoordinate& coord) const;
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

chip_id_t SystemMesh::Impl::get_physical_device_id(const MeshCoordinate& coord) const {
    TT_FATAL(
        coord.dims() == logical_mesh_shape_.dims(),
        "Coordinate dimensions mismatch: {} != {}",
        coord.dims(),
        logical_mesh_shape_.dims());
    for (size_t i = 0; i < coord.dims(); ++i) {
        TT_FATAL(
            coord[i] < logical_mesh_shape_[i],
            "Coordinate at index {} out of bounds; mesh shape {}, coordinate {}",
            i,
            logical_mesh_shape_,
            coord);
    }
    return logical_to_device_id_.at(coord);
}

std::vector<chip_id_t> SystemMesh::Impl::get_mapped_physical_device_ids(const MeshDeviceConfig& config) const {
    std::vector<chip_id_t> physical_device_ids;

    TT_FATAL(
        config.mesh_shape.mesh_size() <= logical_mesh_shape_.mesh_size(),
        "Requested mesh is too big: {}, SystemMesh {}",
        config.mesh_shape.mesh_size(),
        logical_mesh_shape_.mesh_size());

    const size_t system_dimensions = logical_mesh_shape_.dims();

    const MeshCoordinate system_offset = [&config, system_dimensions]() {
        if (config.offset.has_value()) {
            TT_FATAL(
                config.offset->dims() == system_dimensions,
                "Provided offset dimensions mismatch: {} != {}",
                config.offset,
                system_dimensions);
            return *config.offset;
        } else {
            return MeshCoordinate(tt::stl::SmallVector<uint32_t>(system_dimensions, 0));
        }
    }();

    if (is_line_topology(config.mesh_shape)) {
        TT_FATAL(
            std::all_of(system_offset.coords().begin(), system_offset.coords().end(), [](int dim) { return dim == 0; }),
            "Offsets are unsupported for a line mesh");

        // TODO: consider if we can do this in 3D.
        TT_FATAL(logical_mesh_shape_.dims() == 2, "Line topology is only supported for 2D meshes");
        Shape2D shape_2d(logical_mesh_shape_[0], logical_mesh_shape_[1]);

        auto line_length = config.mesh_shape.mesh_size();
        for (const auto& logical_coordinate : MeshDeviceView::get_line_coordinates(line_length, shape_2d)) {
            auto physical_device_id = get_physical_device_id(logical_coordinate);
            physical_device_ids.push_back(physical_device_id);

            log_debug(
                LogMetal, "Logical coordinate: {}, Physical device ID: {}", logical_coordinate, physical_device_id);
        }
        return physical_device_ids;
    }

    TT_FATAL(
        config.mesh_shape.dims() == system_dimensions,
        "Requested mesh shape dimensions mismatch: {} != {}",
        config.mesh_shape,
        logical_mesh_shape_);

    // Attempt to fit the requested mesh into the system mesh, potentially rotating it.
    auto requested_mesh_fits = [this, &system_offset](const tt::stl::SmallVector<uint32_t>& rotated_shape) {
        for (int i = 0; i < logical_mesh_shape_.dims(); ++i) {
            if (system_offset[i] + rotated_shape[i] > logical_mesh_shape_[i]) {
                return false;
            }
        }
        return true;
    };

    tt::stl::SmallVector<uint32_t> rotated_shape(config.mesh_shape.cbegin(), config.mesh_shape.cend());
    size_t rotations = 0;
    while (!requested_mesh_fits(rotated_shape) && rotations < system_dimensions) {
        std::rotate(rotated_shape.begin(), rotated_shape.begin() + 1, rotated_shape.end());
        ++rotations;
    }
    // After rotating N times, no luck. The requested mesh it too big.
    if (rotations == system_dimensions) {
        TT_THROW(
            "Requested mesh is too big and is not rotatable: {} and SystemMesh {}, offset {}",
            config.mesh_shape,
            logical_mesh_shape_,
            system_offset);
    }

    tt::stl::SmallVector<uint32_t> end_coord;
    for (int i = 0; i < system_dimensions; ++i) {
        end_coord.push_back(system_offset[i] + rotated_shape[i] - 1);
    }

    MeshCoordinateRange system_range(system_offset, MeshCoordinate(end_coord));

    for (const auto& system_coord : system_range) {
        auto physical_device_id = get_physical_device_id(system_coord);
        physical_device_ids.push_back(physical_device_id);
        log_debug(LogMetal, "Logical coordinate: {}, Physical device ID: {}", system_coord, physical_device_id);
    }
    return physical_device_ids;
}

std::vector<chip_id_t> SystemMesh::Impl::request_available_devices(const MeshDeviceConfig& config) const {
    log_debug(LogMetal, "Mapping MeshDevice ({})", config.mesh_shape);
    if (config.offset.has_value()) {
        log_debug(LogMetal, "Offset: {}", config.offset.value());
    }

    return config.physical_device_ids.empty() ? this->get_mapped_physical_device_ids(config)
                                              : config.physical_device_ids;
}

SystemMesh::SystemMesh() : pimpl_(std::make_unique<Impl>()) {}

SystemMesh& SystemMesh::instance() {
    static tt::stl::Indestructible<SystemMesh> instance;
    if (!instance.get().pimpl_->is_system_mesh_initialized()) {
        instance.get().pimpl_->initialize();
    }
    return instance.get();
}

chip_id_t SystemMesh::get_physical_device_id(const MeshCoordinate& coord) const {
    return pimpl_->get_physical_device_id(coord);
}

const MeshShape& SystemMesh::get_shape() const { return pimpl_->get_shape(); }

std::vector<chip_id_t> SystemMesh::request_available_devices(const MeshDeviceConfig& config) const {
    return pimpl_->request_available_devices(config);
}

std::vector<chip_id_t> SystemMesh::get_mapped_physical_device_ids(const MeshDeviceConfig& config) const {
    return pimpl_->get_mapped_physical_device_ids(config);
}

}  // namespace tt::tt_metal::distributed
