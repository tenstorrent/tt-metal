// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "submesh_allocator_manager.hpp"

#include "tt_metal/distributed/mesh_device.hpp"

namespace tt::tt_metal::distributed {

uint32_t SubmeshAllocatorDependenciesManager::register_mesh(const MeshDevice& mesh_device) {
    int mesh_id = mesh_device.id();
    auto it = mesh_id_to_allocator_id_.find(mesh_id);
    if (it != mesh_id_to_allocator_id_.end()) {
        return it->second;
    }
    // Single allocator for now.
    uint32_t id = 0;
    mesh_id_to_allocator_id_.emplace(mesh_id, id);
    return id;
}

uint32_t SubmeshAllocatorDependenciesManager::allocator_id_for(const MeshDevice& mesh_device) const {
    int mesh_id = mesh_device.id();
    auto it = mesh_id_to_allocator_id_.find(mesh_id);
    if (it != mesh_id_to_allocator_id_.end()) {
        return it->second;
    }
    // If not found, in single-allocator mode default to 0.
    return 0u;
}

}  // namespace tt::tt_metal::distributed
