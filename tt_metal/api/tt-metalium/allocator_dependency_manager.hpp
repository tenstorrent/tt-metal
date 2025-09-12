// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <tt_stl/strong_type.hpp>
#include <tt_stl/small_vector.hpp>

// Forward declare BankManager's AllocatorID type
namespace tt::tt_metal {
class BankManager;
}

namespace tt::tt_metal::distributed {

class MeshDevice;

/**
 * @brief Manages allocator dependencies between submeshes at the MeshDevice level.
 *
 * This class is responsible for maintaining allocator state and dependencies between
 * parent meshes and submeshes. For now, there is only one allocator state and all
 * AllocatorIDs are set to 0.
 */
class AllocatorDependencyManager {
public:
    // Use the same AllocatorID type as BankManager for consistency
    using AllocatorID = tt::stl::StrongType<uint32_t, struct AllocatorIDTag>;

    /**
     * @brief Construct a new AllocatorDependencyManager with default single allocator
     */
    AllocatorDependencyManager();

    /**
     * @brief Get the AllocatorID for a given mesh device
     *
     * For now, always returns AllocatorID{0} since we only have one allocator state.
     *
     * @param mesh_device The mesh device to get allocator ID for
     * @return AllocatorID The allocator ID (currently always 0)
     */
    AllocatorID get_allocator_id(const MeshDevice* mesh_device) const;

    /**
     * @brief Register a new submesh with the dependency manager
     *
     * @param submesh The submesh to register
     * @param parent_mesh The parent mesh device
     */
    void register_submesh(std::shared_ptr<MeshDevice> submesh, std::shared_ptr<MeshDevice> parent_mesh);

    /**
     * @brief Unregister a submesh from the dependency manager
     *
     * @param submesh The submesh to unregister
     */
    void unregister_submesh(std::shared_ptr<MeshDevice> submesh);

    /**
     * @brief Get all submeshes registered with this manager
     *
     * @return std::vector<std::weak_ptr<MeshDevice>> Vector of weak pointers to submeshes
     */
    std::vector<std::weak_ptr<MeshDevice>> get_submeshes() const;

    /**
     * @brief Get the total number of allocators managed
     *
     * @return uint32_t Number of allocators (currently always 1)
     */
    uint32_t get_num_allocators() const;

private:
    // For now, we only have one allocator state
    static constexpr uint32_t NUM_ALLOCATORS = 1;
    static const AllocatorID DEFAULT_ALLOCATOR_ID;

    // Track submeshes (weak pointers to avoid circular dependencies)
    std::vector<std::weak_ptr<MeshDevice>> submeshes_;

    // Track parent-child relationships for future dependency management
    std::unordered_map<MeshDevice*, std::weak_ptr<MeshDevice>> submesh_to_parent_;
};

}  // namespace tt::tt_metal::distributed
