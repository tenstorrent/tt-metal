// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/allocator_dependency_manager.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <algorithm>
#include <cstdint>

namespace tt::tt_metal::distributed {

// Define the static constant
const AllocatorDependencyManager::AllocatorID AllocatorDependencyManager::DEFAULT_ALLOCATOR_ID{0};

AllocatorDependencyManager::AllocatorDependencyManager() {
    // Initialize with default state - single allocator
    submeshes_.clear();
    submesh_to_parent_.clear();
}

AllocatorDependencyManager::AllocatorID AllocatorDependencyManager::get_allocator_id(
    const MeshDevice* mesh_device) const {
    // For now, always return allocator ID 0 since we only have one allocator state
    (void)mesh_device;  // Suppress unused parameter warning
    return DEFAULT_ALLOCATOR_ID;
}

void AllocatorDependencyManager::register_submesh(
    std::shared_ptr<MeshDevice> submesh, std::shared_ptr<MeshDevice> parent_mesh) {
    if (submesh) {
        submeshes_.push_back(submesh);
        if (parent_mesh) {
            submesh_to_parent_[submesh.get()] = parent_mesh;
        }
    }
}

void AllocatorDependencyManager::unregister_submesh(std::shared_ptr<MeshDevice> submesh) {
    if (!submesh) {
        return;
    }

    // Remove from submeshes list
    submeshes_.erase(
        std::remove_if(
            submeshes_.begin(),
            submeshes_.end(),
            [&submesh](const std::weak_ptr<MeshDevice>& weak_submesh) {
                return weak_submesh.expired() || weak_submesh.lock() == submesh;
            }),
        submeshes_.end());

    // Remove from parent mapping
    submesh_to_parent_.erase(submesh.get());
}

std::vector<std::weak_ptr<MeshDevice>> AllocatorDependencyManager::get_submeshes() const {
    // Clean up expired weak pointers and return current submeshes
    std::vector<std::weak_ptr<MeshDevice>> active_submeshes;
    for (const auto& weak_submesh : submeshes_) {
        if (!weak_submesh.expired()) {
            active_submeshes.push_back(weak_submesh);
        }
    }
    return active_submeshes;
}

uint32_t AllocatorDependencyManager::get_num_allocators() const { return NUM_ALLOCATORS; }

}  // namespace tt::tt_metal::distributed
