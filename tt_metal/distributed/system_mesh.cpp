// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/container/vector.hpp>
#include <stdint.h>
#include <system_mesh.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt_stl/indestructible.hpp>
#include <algorithm>
#include <cstddef>
#include <unordered_set>

#include "assert.hpp"
#include <tt-logger/tt-logger.hpp>
#include "mesh_config.hpp"
#include "mesh_coord.hpp"
#include "shape_base.hpp"
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>
#include "tt_metal/distributed/coordinate_translation.hpp"
#include "tt_metal/distributed/coordinate_translator.hpp"

#include "impl/context/metal_context.hpp"
#include <tt-metalium/control_plane.hpp>

namespace tt::tt_metal::distributed {

class SystemMesh::Impl {
private:
    MeshShape global_shape_;
    MeshContainer<MaybeRemote<PhysicalMeshCoordinate>> physical_coordinates_;
    MeshCoordinate local_offset_;

public:
    Impl();

    const MeshShape& shape() const;
    const MeshShape& local_shape() const;
    MeshCoordinate get_global_device_coordinate(int physical_device_id) const;
    std::vector<MaybeRemoteDeviceId> get_mapped_physical_device_ids(
        const MeshShape& shape, const std::optional<MeshCoordinate>& offset = std::nullopt) const;
    chip_id_t get_physical_device_id(const MeshCoordinate& coord) const;
    uint32_t get_physical_mesh_id(const MeshCoordinate& coord) const;
    bool is_local_coordinate(const MeshCoordinate& coord) const;
};

// Implementation of public methods
SystemMesh::Impl::Impl() 
    : global_shape_(tt::tt_metal::MetalContext::instance().get_control_plane().get_physical_mesh_shape(tt::tt_metal::MetalContext::instance().get_control_plane().get_local_mesh_id_bindings()[0], tt::tt_fabric::MeshScope::GLOBAL)),
      physical_coordinates_(global_shape_, MaybeRemote<PhysicalMeshCoordinate>::remote()),  // Initialize all as remote
      local_offset_(tt::tt_metal::MetalContext::instance().get_control_plane().get_local_mesh_offset()) {
    // Get local physical coordinates
    auto local_coordinates = get_system_mesh_coordinate_translation_map();
    CoordinateTranslator translator(local_coordinates.shape(), local_offset_);
    
    // Map local coordinates to global mesh
    for (const auto& [local_coord, physical_coord] : local_coordinates) {
        auto global_coord = translator.local_to_global(local_coord);
        auto& coord_range = physical_coordinates_.coord_range();
        if (coord_range.contains(global_coord)) {
            physical_coordinates_.at(global_coord) = MaybeRemote<PhysicalMeshCoordinate>::local(physical_coord);
        }
    }
}

bool SystemMesh::Impl::is_local_coordinate(const MeshCoordinate& coord) const {
    if (!physical_coordinates_.coord_range().contains(coord)) {
        return false;
    }
    return physical_coordinates_.at(coord).is_local();
}

const MeshShape& SystemMesh::Impl::shape() const { return global_shape_; }

const MeshShape& SystemMesh::Impl::local_shape() const { 
    auto local_coordinates = get_system_mesh_coordinate_translation_map();
    return local_coordinates.shape();
}

chip_id_t SystemMesh::Impl::get_physical_device_id(const MeshCoordinate& coord) const {
    TT_FATAL(is_local_coordinate(coord), "Coordinate {} is not in the local mesh", coord);
    
    const auto& maybe_physical = physical_coordinates_.at(coord);
    auto physical_device_id = maybe_physical.value().chip_id();
    log_debug(LogDistributed, "Global coordinate: {} mapped to physical device ID: {}", 
              coord, physical_device_id);
    return physical_device_id;
}

uint32_t SystemMesh::Impl::get_physical_mesh_id(const MeshCoordinate& coord) const {
    TT_FATAL(is_local_coordinate(coord), "Coordinate {} is not in the local mesh", coord);
    const auto& maybe_physical = physical_coordinates_.at(coord);
    return *maybe_physical.value().mesh_id();
}

MeshCoordinate SystemMesh::Impl::get_global_device_coordinate(int physical_device_id) const {
    for (const auto& [global_coordinate, maybe_physical] : physical_coordinates_) {
        if (maybe_physical.is_local()) {
            const auto& physical_mesh_coordinate = maybe_physical.value();
            if (physical_mesh_coordinate.chip_id() == physical_device_id) {
                return global_coordinate;
            }
        }
    }
    TT_THROW("Physical device ID {} not found in the system mesh", physical_device_id);
}

std::vector<MaybeRemoteDeviceId> SystemMesh::Impl::get_mapped_physical_device_ids(
    const MeshShape& shape, const std::optional<MeshCoordinate>& offset) const {
    std::vector<MaybeRemoteDeviceId> physical_device_ids;

    const MeshShape& system_shape = this->shape();
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

    if (shape.is_line_topology()) {
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
            if (is_local_coordinate(logical_coordinate)) {
                auto physical_device_id = get_physical_device_id(logical_coordinate);
                log_debug(
                    LogDistributed, "Logical coordinate: {}, Physical device ID: {}", logical_coordinate, physical_device_id);
                physical_device_ids.push_back(MaybeRemoteDeviceId::local(physical_device_id));
            } else {
                log_debug(LogDistributed, "Logical coordinate: {} is remote", logical_coordinate);
                physical_device_ids.push_back(MaybeRemoteDeviceId::remote());
            }
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
                if (is_local_coordinate(system_coord)) {
                    auto physical_device_id = get_physical_device_id(system_coord);
                    log_debug(LogDistributed, "Logical coordinate: {}, Physical device ID: {}", system_coord, physical_device_id);
                    physical_device_ids.push_back(MaybeRemoteDeviceId::local(physical_device_id));
                } else {
                    log_debug(LogDistributed, "Logical coordinate: {} is remote", system_coord);
                    physical_device_ids.push_back(MaybeRemoteDeviceId::remote());
                }
            }
        }
    } else {
        for (const auto& system_coord : system_range) {
            if (is_local_coordinate(system_coord)) {
                auto physical_device_id = get_physical_device_id(system_coord);
                physical_device_ids.push_back(MaybeRemoteDeviceId::local(physical_device_id));
                log_debug(LogMetal, "Logical coordinate: {}, Physical device ID: {}", system_coord, physical_device_id);
            } else {
                physical_device_ids.push_back(MaybeRemoteDeviceId::remote());
                log_debug(LogMetal, "Logical coordinate: {} is remote", system_coord);
            }
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

const MeshShape& SystemMesh::shape() const { return pimpl_->shape(); }

const MeshShape& SystemMesh::local_shape() const { return pimpl_->local_shape(); }

MeshCoordinate SystemMesh::get_global_device_coordinate(int physical_device_id) const {
    return pimpl_->get_global_device_coordinate(physical_device_id);
}

std::vector<MaybeRemoteDeviceId> SystemMesh::get_mapped_physical_device_ids(
    const MeshShape& shape, const std::optional<MeshCoordinate>& offset) const {
    return pimpl_->get_mapped_physical_device_ids(shape, offset);
}

}  // namespace tt::tt_metal::distributed
