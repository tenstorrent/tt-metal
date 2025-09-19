// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <cstdint>
#include <vector>
#include <memory>
#include "tt_metal/api/tt-metalium/mesh_coord.hpp"
#include "tt_metal/impl/allocator/bank_manager.hpp"

namespace tt::tt_metal::distributed {

class MeshDevice;

// Manages allocator IDs and dependencies across a parent mesh and its submeshes.
// Assigns state IDs to MeshCoordinateRanges and creates AllocatorDependencies based on overlap.
class SubmeshManager {
public:
    using StateID = uint32_t;
    using AllocatorDependencies = tt::tt_metal::BankManager::AllocatorDependencies;
    using AllocatorID = tt::tt_metal::BankManager::AllocatorDependencies::AllocatorID;

    SubmeshManager() = default;

    // Constructor that takes ranges and creates dependencies internally
    // Assigns state IDs to the given MeshCoordinateRanges and creates AllocatorDependencies
    // based on overlap criteria: if two ranges overlap on any coordinate, they are dependent
    explicit SubmeshManager(const std::vector<MeshCoordinateRange>& ranges);

    // Get the stored dependencies
    const AllocatorDependencies& get_dependencies() const { return dependencies_; }

    // Get the state ID assigned to a specific range
    StateID get_state_id(const MeshCoordinateRange& range) const;

    // Get all assigned state IDs
    const std::vector<StateID>& get_state_ids() const { return state_ids_; }

private:
    std::vector<MeshCoordinateRange> ranges_;
    std::vector<StateID> state_ids_;
    std::unordered_map<MeshCoordinateRange, StateID> range_to_state_id_;
    AllocatorDependencies dependencies_;
};

}  // namespace tt::tt_metal::distributed
