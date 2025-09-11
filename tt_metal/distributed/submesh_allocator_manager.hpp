// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <cstdint>

namespace tt::tt_metal::distributed {

class MeshDevice;

// Manages allocator IDs and dependencies across a parent mesh and its submeshes.
// For now: single allocator state across all meshes; all AllocatorIDs are 0.
class SubmeshAllocatorDependenciesManager {
public:
    using AllocatorID = uint32_t;

    SubmeshAllocatorDependenciesManager() = default;

    // Register a mesh or submesh; returns its AllocatorID.
    // Currently always returns 0 for single-allocator mode.
    AllocatorID register_mesh(const MeshDevice& mesh_device);

    // Get the AllocatorID for a mesh.
    AllocatorID allocator_id_for(const MeshDevice& mesh_device) const;

private:
    // Map MeshDevice id -> AllocatorID
    std::unordered_map<int, AllocatorID> mesh_id_to_allocator_id_;
};

}  // namespace tt::tt_metal::distributed
