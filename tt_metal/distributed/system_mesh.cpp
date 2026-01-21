// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <system_mesh.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt_stl/indestructible.hpp>
#include <algorithm>
#include <cstddef>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include "mesh_config.hpp"
#include "mesh_coord.hpp"
#include "shape_base.hpp"
#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>
#include "tt_metal/distributed/system_mesh_translation_map.hpp"
#include "tt_metal/distributed/distributed_coordinate_translator.hpp"

#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt::tt_metal::distributed {
// Helper type to keep track of device ID and fabric node ID for a given mesh coordinate.
struct MappedDevice {
    MaybeRemote<int> device_id;
    tt::tt_fabric::FabricNodeId fabric_node_id;
};

namespace {
// Initializes a mesh container with MappedDevice objects, with configured fabric node IDs.
MeshContainer<MappedDevice> initialize_mapped_devices(const tt::tt_fabric::MeshId mesh_id, const MeshShape& shape) {
    std::vector<MappedDevice> system_mesh_devices;
    system_mesh_devices.reserve(shape.mesh_size());
    for (int linear_index = 0; linear_index < shape.mesh_size(); ++linear_index) {
        system_mesh_devices.push_back(MappedDevice{
            .device_id = MaybeRemote<int>::remote(),
            .fabric_node_id = tt::tt_fabric::FabricNodeId(mesh_id, linear_index)});
    }
    return MeshContainer<MappedDevice>(shape, std::move(system_mesh_devices));
}

}  // namespace
class SystemMesh::Impl {
private:
    tt::tt_fabric::MeshId mesh_id_;
    DistributedCoordinateTranslator coordinate_translator_;
    MeshContainer<MappedDevice> system_mapped_devices_;

    MappedDevice get_system_mapped_device(const MeshCoordinate& coord) const;

public:
    Impl();

    const DistributedCoordinateTranslator& coordinate_translator() const;

    MappedDevices get_mapped_devices(
        const std::optional<MeshShape>& shape, const std::optional<MeshCoordinate>& offset = std::nullopt) const;
};

MappedDevice SystemMesh::Impl::get_system_mapped_device(const MeshCoordinate& coord) const {
    auto system_mapped_device = system_mapped_devices_.at(coord);
    if (system_mapped_device.device_id.is_local()) {
        log_debug(
            LogDistributed,
            "Mesh coordinate: {} is local, Physical device ID: {}, Fabric node ID: {}",
            coord,
            *system_mapped_device.device_id,
            system_mapped_device.fabric_node_id);
    } else {
        log_debug(
            LogDistributed,
            "Mesh coordinate: {} is remote, Fabric node ID: {}",
            coord,
            system_mapped_device.fabric_node_id);
    }

    return system_mapped_device;
}

// Implementation of public methods
SystemMesh::Impl::Impl() :
    mesh_id_(MetalContext::instance().get_control_plane().get_local_mesh_id_bindings()[0]),
    coordinate_translator_(
        MetalContext::instance().get_control_plane().get_physical_mesh_shape(
            mesh_id_,  //
            tt::tt_fabric::MeshScope::GLOBAL),
        MetalContext::instance().get_control_plane().get_physical_mesh_shape(
            mesh_id_,  //
            tt::tt_fabric::MeshScope::LOCAL),
        MetalContext::instance().get_control_plane().get_local_mesh_offset()),
    system_mapped_devices_(initialize_mapped_devices(mesh_id_, coordinate_translator_.global_shape())) {
    log_debug(
        LogDistributed,
        "SystemMesh: Global shape: {}, Local shape: {}, Local offset: {}",
        coordinate_translator_.global_shape(),
        coordinate_translator_.local_shape(),
        coordinate_translator_.local_offset());

    // Get local physical coordinates
    const auto& local_physical_translation_map = get_system_mesh_coordinate_translation_map();
    TT_FATAL(
        local_physical_translation_map.shape() == coordinate_translator_.local_shape(),
        "Local coordinates shape mismatch: {} != {}",
        local_physical_translation_map.shape(),
        coordinate_translator_.local_shape());

    // Populate chip IDs for host-local devices.
    for (const auto& local_coord : MeshCoordinateRange(coordinate_translator_.local_shape())) {
        TT_FATAL(
            local_physical_translation_map.at(local_coord).mesh_id() == mesh_id_,
            "Mesh id mismatch for coordinate {}: {} != {}",
            local_coord,
            local_physical_translation_map.at(local_coord).mesh_id(),
            mesh_id_);

        const auto global_coord = coordinate_translator_.local_to_global(local_coord);
        log_debug(
            LogDistributed,
            "SystemMesh: Populating global coordinate {} with physical coordinate {} at local coordinate {}",
            global_coord,
            local_physical_translation_map.at(local_coord),
            local_coord);
        system_mapped_devices_.at(global_coord).device_id =
            MaybeRemote<int>::local(local_physical_translation_map.at(local_coord).chip_id());
    }
}

const DistributedCoordinateTranslator& SystemMesh::Impl::coordinate_translator() const {
    return coordinate_translator_;
}

SystemMesh::MappedDevices SystemMesh::Impl::get_mapped_devices(
    const std::optional<MeshShape>& shape, const std::optional<MeshCoordinate>& offset) const {
    MappedDevices mapped_devices;

    const MeshShape& system_shape = coordinate_translator_.global_shape();
    const MeshShape requested_shape = shape.value_or(system_shape);
    mapped_devices.mesh_shape = requested_shape;
    const size_t system_dimensions = system_shape.dims();

    const MeshCoordinate system_offset = [&offset, system_dimensions]() {
        if (offset.has_value()) {
            TT_FATAL(
                offset->dims() == system_dimensions,
                "Provided offset dimensions mismatch: {} != {}",
                offset,
                system_dimensions);
            return *offset;
        }
        return MeshCoordinate::zero_coordinate(system_dimensions);
    }();

    if (requested_shape.is_line_topology()) {
        // TODO: consider if we can do this in 3D.
        TT_FATAL(system_shape.dims() == 2, "Line topology is only supported for 2D meshes");
        TT_FATAL(
            system_shape[0] > system_offset[0] && system_shape[1] > system_offset[1],
            "The specified offset {} is out of bounds for the system mesh shape {}",
            system_offset,
            system_shape);
        Shape2D system_mesh_2d(system_shape[0], system_shape[1]);
        Shape2D system_offset_2d(system_offset[0], system_offset[1]);

        auto line_length = requested_shape.mesh_size();
        for (const auto& logical_coordinate :
             MeshDeviceView::get_line_coordinates(line_length, system_mesh_2d, system_offset_2d)) {
            const auto mapped_device = get_system_mapped_device(logical_coordinate);
            mapped_devices.device_ids.push_back(mapped_device.device_id);
            mapped_devices.fabric_node_ids.push_back(mapped_device.fabric_node_id);
        }
        return mapped_devices;
    }

    TT_FATAL(
        requested_shape.dims() == system_dimensions,
        "Requested mesh shape dimensions mismatch: {} != {}",
        requested_shape,
        system_shape);

    // Attempt to fit the requested mesh into the system mesh, potentially rotating it.
    auto requested_mesh_fits = [&system_offset, &system_shape](const tt::stl::SmallVector<uint32_t>& rotated_shape) {
        for (int i = 0; i < system_shape.dims(); ++i) {
            if (system_offset[i] + rotated_shape[i] > system_shape[i]) {
                return false;
            }
        }
        return true;
    };

    tt::stl::SmallVector<uint32_t> rotated_shape(requested_shape.cbegin(), requested_shape.cend());
    size_t rotations = 0;
    while (!requested_mesh_fits(rotated_shape) && rotations < system_dimensions) {
        std::rotate(rotated_shape.begin(), rotated_shape.begin() + 1, rotated_shape.end());
        ++rotations;
    }
    // After rotating N times, no luck. The requested mesh it too big.
    if (rotations == system_dimensions) {
        TT_THROW(
            "Requested mesh is too big and is not rotatable: {} and SystemMesh {}, offset {}",
            requested_shape,
            system_shape,
            system_offset);
    }

    tt::stl::SmallVector<uint32_t> end_coord;
    for (int i = 0; i < system_dimensions; ++i) {
        end_coord.push_back(system_offset[i] + rotated_shape[i] - 1);
    }

    MeshCoordinateRange system_range(system_offset, MeshCoordinate(end_coord));

    // Iterate over the system mesh and map the logical coordinates to system mesh devices.
    bool is_rotated = rotations > 0;  // Track if we rotated the mesh.
    if (is_rotated) {
        TT_FATAL(rotations == 1 and system_shape.dims() == 2, "Mesh rotation is only supported for 2D meshes");

        // Iterate through user-requested shape, transposing the rows and columns
        for (int i = 0; i < requested_shape[0]; i++) {
            for (int j = 0; j < requested_shape[1]; j++) {
                const auto system_coord = MeshCoordinate(j, i);
                const auto mapped_device = get_system_mapped_device(system_coord);
                mapped_devices.device_ids.push_back(mapped_device.device_id);
                mapped_devices.fabric_node_ids.push_back(mapped_device.fabric_node_id);
            }
        }
    } else {
        for (const auto& system_coord : system_range) {
            const auto mapped_device = get_system_mapped_device(system_coord);
            mapped_devices.device_ids.push_back(mapped_device.device_id);
            mapped_devices.fabric_node_ids.push_back(mapped_device.fabric_node_id);
        }
    }

    return mapped_devices;
}

SystemMesh::SystemMesh() : pimpl_(std::make_unique<Impl>()) {}

SystemMesh& SystemMesh::instance() {
    static tt::stl::Indestructible<SystemMesh> instance;
    return instance.get();
}

const MeshShape& SystemMesh::shape() const { return pimpl_->coordinate_translator().global_shape(); }
const MeshShape& SystemMesh::local_shape() const { return pimpl_->coordinate_translator().local_shape(); }

SystemMesh::MappedDevices SystemMesh::get_mapped_devices(
    const std::optional<MeshShape>& shape, const std::optional<MeshCoordinate>& offset) const {
    return pimpl_->get_mapped_devices(shape, offset);
}

}  // namespace tt::tt_metal::distributed
