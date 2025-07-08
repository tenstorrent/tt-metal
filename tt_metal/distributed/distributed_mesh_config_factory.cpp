// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/distributed_mesh_config_factory.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::distributed {

DistributedMeshConfig DistributedMeshConfigFactory::create_from_control_plane(const MeshDeviceConfig& config) {
    auto local_info = fetch_local_mesh_info();
    
    log_debug(LogDistributed, "[DistributedMeshConfigFactory] Creating config - Global shape: {}, Local shape: {}, Local offset: {}", 
              config.mesh_shape(), local_info.local_shape, local_info.local_offset);
    
    return DistributedMeshConfig(
        config.mesh_shape(),
        local_info.local_shape,
        local_info.local_offset
    );
}

DistributedMeshConfigFactory::LocalMeshInfo DistributedMeshConfigFactory::fetch_local_mesh_info() {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    
    LocalMeshInfo info{
        .local_shape = control_plane.get_physical_mesh_shape(control_plane.get_local_mesh_id_bindings()[0], tt::tt_fabric::MeshScope::LOCAL),
        .local_offset = control_plane.get_local_mesh_offset()
    };
    
    return info;
}

}  // namespace tt::tt_metal::distributed