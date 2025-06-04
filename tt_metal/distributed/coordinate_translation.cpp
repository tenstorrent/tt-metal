// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/coordinate_translation.hpp"

#include <boost/move/utility_core.hpp>
#include <unordered_set>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "control_plane.hpp"
#include <tt_stl/indestructible.hpp>
#include "logger.hpp"
#include "mesh_coord.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/cluster_descriptor_types.h>

namespace tt::tt_metal::distributed {

const MeshContainer<PhysicalMeshCoordinate>& get_system_mesh_coordinate_translation_map() {
    static tt::stl::Indestructible<MeshContainer<PhysicalMeshCoordinate>> kTranslationMap([]() {
        const auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
        TT_FATAL(control_plane != nullptr, "Control plane must be initialized before MeshDevice can be created.");

        const auto mesh_ids = control_plane->get_user_physical_mesh_ids();
        TT_FATAL(!mesh_ids.empty(), "There are no user physical meshes in the system found by control plane.");

        if (mesh_ids.size() > 1) {
            tt::log_warning(LogMetal, "Only one user physical mesh is supported, using the first one");
        }

        const auto mesh_id = mesh_ids.front();
        const auto mesh_shape = control_plane->get_physical_mesh_shape(mesh_id);

        // Validate that the physical chip ids are unique.
        std::unordered_set<chip_id_t> unique_chip_ids;

        std::vector<PhysicalMeshCoordinate> physical_coordinates;
        physical_coordinates.reserve(mesh_shape.mesh_size());
        for (int logical_chip_id = 0; logical_chip_id < mesh_shape.mesh_size(); ++logical_chip_id) {
            // Query the control plane to get the physical chip id from logical chip id
            const auto physical_chip_id = control_plane->get_physical_chip_id_from_fabric_node_id(
                tt::tt_fabric::FabricNodeId(mesh_id, logical_chip_id));
            TT_FATAL(
                unique_chip_ids.insert(physical_chip_id).second,
                "Found duplicate physical chip id: {}, mesh id: {}",
                physical_chip_id,
                mesh_id);
            physical_coordinates.push_back(PhysicalMeshCoordinate(/*mesh_id=*/*mesh_id, /*chip_id=*/physical_chip_id));
        }
        return MeshContainer<PhysicalMeshCoordinate>(mesh_shape, std::move(physical_coordinates));
    }());
    return kTranslationMap.get();
}

}  // namespace tt::tt_metal::distributed
