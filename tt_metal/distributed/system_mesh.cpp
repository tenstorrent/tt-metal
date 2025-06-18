// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/container/vector.hpp>
#include <stdint.h>
#include <system_mesh.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt_stl/indestructible.hpp>
#include <algorithm>
#include <cstddef>

#include "assert.hpp"
#include <tt-logger/tt-logger.hpp>
#include "mesh_config.hpp"
#include "mesh_coord.hpp"
#include "shape_base.hpp"
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>
#include "tt_metal/distributed/coordinate_translation.hpp"

namespace tt::tt_metal::distributed {

class SystemMesh::Impl {
private:
    MeshContainer<PhysicalMeshCoordinate> physical_coordinates_;

public:
    Impl();

    const MeshShape& get_shape() const;
    MeshCoordinate get_global_device_coordinate(int physical_device_id) const;
    std::vector<chip_id_t> get_mapped_physical_device_ids(
        const MeshShape& shape, const std::optional<MeshCoordinate>& offset = std::nullopt) const;
    chip_id_t get_physical_device_id(const MeshCoordinate& coord) const;
    uint32_t get_physical_mesh_id(const MeshCoordinate& coord) const;
};

// Implementation of public methods
SystemMesh::Impl::Impl() : physical_coordinates_(get_system_mesh_coordinate_translation_map()) {
    for (const auto& [logical_coordinate, physical_mesh_coordinate] : physical_coordinates_) {
        log_debug(
            LogMetal,
            "Logical Coordinate: ({}, {}), Physical Mesh Coordinate: (Mesh ID {}, Chip ID {})",
            logical_coordinate[0],
            logical_coordinate[1],
            physical_mesh_coordinate.mesh_id(),
            physical_mesh_coordinate.chip_id());
    }
}

const MeshShape& SystemMesh::Impl::get_shape() const { return physical_coordinates_.shape(); }

chip_id_t SystemMesh::Impl::get_physical_device_id(const MeshCoordinate& coord) const {
    return physical_coordinates_.at(coord).chip_id();
}

uint32_t SystemMesh::Impl::get_physical_mesh_id(const MeshCoordinate& coord) const {
    return *physical_coordinates_.at(coord).mesh_id();
}

MeshCoordinate SystemMesh::Impl::get_global_device_coordinate(int physical_device_id) const {
    for (const auto& [logical_coordinate, physical_mesh_coordinate] : physical_coordinates_) {
        if (physical_mesh_coordinate.chip_id() == physical_device_id) {
            return logical_coordinate;
        }
    }
    TT_THROW("Physical device ID {} not found in the system mesh", physical_device_id);
}

std::vector<chip_id_t> SystemMesh::Impl::get_mapped_physical_device_ids(
    const MeshShape& shape, const std::optional<MeshCoordinate>& offset) const {
    std::vector<chip_id_t> physical_device_ids;

    const MeshShape& system_shape = this->get_shape();
    TT_FATAL(
        shape.mesh_size() <= system_shape.mesh_size(),
        "Requested mesh is too big: {}, SystemMesh {}",
        shape.mesh_size(),
        system_shape.mesh_size());

    const size_t system_dimensions = system_shape.dims();

    const MeshCoordinate system_offset = [&offset, system_dimensions]() {
        if (offset.has_value()) {
            TT_FATAL(
                offset->dims() == system_dimensions,
                "Provided offset dimensions mismatch: {} != {}",
                offset,
                system_dimensions);
            return *offset;
        } else {
            return MeshCoordinate::zero_coordinate(system_dimensions);
        }
    }();

    if (is_line_topology(shape)) {
        // TODO: consider if we can do this in 3D.
        TT_FATAL(system_shape.dims() == 2, "Line topology is only supported for 2D meshes");
        TT_FATAL(
            system_shape[0] > system_offset[0] && system_shape[1] > system_offset[1],
            "The specifed offset {} is out of bounds for the system mesh shape {}",
            system_offset,
            system_shape);
        Shape2D system_mesh_2d(system_shape[0], system_shape[1]);
        Shape2D system_offset_2d(system_offset[0], system_offset[1]);

        auto line_length = shape.mesh_size();
        for (const auto& logical_coordinate :
             MeshDeviceView::get_line_coordinates(line_length, system_mesh_2d, system_offset_2d)) {
            auto physical_device_id = get_physical_device_id(logical_coordinate);
            physical_device_ids.push_back(physical_device_id);

            log_debug(
                LogMetal, "Logical coordinate: {}, Physical device ID: {}", logical_coordinate, physical_device_id);
        }
        return physical_device_ids;
    }

    TT_FATAL(
        shape.dims() == system_dimensions, "Requested mesh shape dimensions mismatch: {} != {}", shape, system_shape);

    // Attempt to fit the requested mesh into the system mesh, potentially rotating it.
    auto requested_mesh_fits =
        [this, &system_offset, &system_shape](const tt::stl::SmallVector<uint32_t>& rotated_shape) {
            for (int i = 0; i < system_shape.dims(); ++i) {
                if (system_offset[i] + rotated_shape[i] > system_shape[i]) {
                    return false;
                }
            }
            return true;
        };

    tt::stl::SmallVector<uint32_t> rotated_shape(shape.cbegin(), shape.cend());
    size_t rotations = 0;
    while (!requested_mesh_fits(rotated_shape) && rotations < system_dimensions) {
        std::rotate(rotated_shape.begin(), rotated_shape.begin() + 1, rotated_shape.end());
        ++rotations;
    }
    // After rotating N times, no luck. The requested mesh it too big.
    if (rotations == system_dimensions) {
        TT_THROW(
            "Requested mesh is too big and is not rotatable: {} and SystemMesh {}, offset {}",
            shape,
            system_shape,
            system_offset);
    }

    tt::stl::SmallVector<uint32_t> end_coord;
    for (int i = 0; i < system_dimensions; ++i) {
        end_coord.push_back(system_offset[i] + rotated_shape[i] - 1);
    }

    MeshCoordinateRange system_range(system_offset, MeshCoordinate(end_coord));

    // Iterate over the system mesh and map the logical coordinates to physical device IDs.
    bool is_rotated = rotations > 0;  // Track if we rotated the mesh.
    if (is_rotated) {
        TT_FATAL(rotations == 1 and system_shape.dims() == 2, "Mesh rotation is only supported for 2D meshes");

        // Iterate through user-requested shape, transposing the rows and columns
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                auto system_coord = MeshCoordinate(j, i);
                auto physical_device_id = get_physical_device_id(system_coord);
                physical_device_ids.push_back(physical_device_id);
                log_debug(LogMetal, "Logical coordinate: {}, Physical device ID: {}", system_coord, physical_device_id);
            }
        }
    } else {
        for (const auto& system_coord : system_range) {
            auto physical_device_id = get_physical_device_id(system_coord);
            physical_device_ids.push_back(physical_device_id);
            log_debug(LogMetal, "Logical coordinate: {}, Physical device ID: {}", system_coord, physical_device_id);
        }
    }

    return physical_device_ids;
}

SystemMesh::SystemMesh() : pimpl_(std::make_unique<Impl>()) {}

SystemMesh& SystemMesh::instance() {
    static tt::stl::Indestructible<SystemMesh> instance;
    return instance.get();
}

chip_id_t SystemMesh::get_physical_device_id(const MeshCoordinate& coord) const {
    return pimpl_->get_physical_device_id(coord);
}

uint32_t SystemMesh::get_physical_mesh_id(const MeshCoordinate& coord) const {
    return pimpl_->get_physical_mesh_id(coord);
}

const MeshShape& SystemMesh::get_shape() const { return pimpl_->get_shape(); }

MeshCoordinate SystemMesh::get_global_device_coordinate(int physical_device_id) const {
    return pimpl_->get_global_device_coordinate(physical_device_id);
}

std::vector<chip_id_t> SystemMesh::get_mapped_physical_device_ids(
    const MeshShape& shape, const std::optional<MeshCoordinate>& offset) const {
    return pimpl_->get_mapped_physical_device_ids(shape, offset);
}

}  // namespace tt::tt_metal::distributed
