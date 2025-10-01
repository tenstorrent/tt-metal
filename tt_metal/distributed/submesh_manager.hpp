// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>

#include "tt_metal/api/tt-metalium/mesh_coord.hpp"
#include "tt_metal/impl/allocator/bank_manager.hpp"

namespace tt::tt_metal::distributed {

// Bundles a shared submesh manager with a per-submesh state identifier
class SubmeshState {
public:
    using StateID = uint32_t;

    // Internal manager that used to be exposed as SubmeshManager
    class Manager {
    public:
        using AllocatorDependencies = tt::tt_metal::BankManager::AllocatorDependencies;
        using AllocatorID = tt::tt_metal::BankManager::AllocatorDependencies::AllocatorID;

        Manager() = default;
        explicit Manager(const std::vector<MeshCoordinateRange>& ranges);

        const AllocatorDependencies& get_dependencies() const;

        StateID get_state_id(const MeshCoordinateRange& range) const;

    private:
        std::unordered_map<MeshCoordinateRange, StateID> range_to_state_id_;
        AllocatorDependencies dependencies_;
    };

    SubmeshState(std::shared_ptr<Manager> manager, StateID id) : manager_(std::move(manager)), id_(id) {}

    const std::shared_ptr<Manager>& manager() const { return manager_; }
    StateID id() const { return id_; }
    const Manager::AllocatorDependencies& get_dependencies() const { return manager_->get_dependencies(); }

    // Convenience: create one shared manager and a SubmeshState per input range
    static std::vector<std::shared_ptr<SubmeshState>> create_states_for_ranges(
        const std::vector<MeshCoordinateRange>& ranges);

private:
    std::shared_ptr<Manager> manager_;
    StateID id_ = 0;
};

}  // namespace tt::tt_metal::distributed
