// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "submesh_manager.hpp"
#include "mesh_device.hpp"
#include <functional>
#include <algorithm>
#include <unordered_set>
#include <cstdint>
#include "assert.hpp"
#include <tt_stl/small_vector.hpp>

namespace tt::tt_metal::distributed {

SubmeshManager::SubmeshManager(const std::vector<MeshCoordinateRange>& ranges) {
    // Assign state IDs to each range
    ranges_ = ranges;
    state_ids_.reserve(ranges.size());

    for (size_t i = 0; i < ranges.size(); ++i) {
        StateID state_id = static_cast<StateID>(i);
        state_ids_.push_back(state_id);
        range_to_state_id_[ranges[i]] = state_id;
    }

    // Build dependency map based on overlap criteria
    std::unordered_map<AllocatorID, ttsl::SmallVector<AllocatorID>> dependencies_map;

    for (size_t i = 0; i < ranges.size(); ++i) {
        AllocatorID allocator_i{static_cast<uint32_t>(i)};
        ttsl::SmallVector<AllocatorID> dependencies;

        for (size_t j = 0; j < ranges.size(); ++j) {
            if (i != j) {
                AllocatorID allocator_j{static_cast<uint32_t>(j)};

                // Check if ranges overlap on any coordinate
                if (ranges[i].intersects(ranges[j])) {
                    dependencies.push_back(allocator_j);
                }
            }
        }

        if (!dependencies.empty()) {
            dependencies_map[allocator_i] = std::move(dependencies);
        }
    }

    dependencies_ = tt::tt_metal::BankManager::AllocatorDependencies(dependencies_map);
}

SubmeshManager::StateID SubmeshManager::get_state_id(const MeshCoordinateRange& range) const {
    auto it = range_to_state_id_.find(range);
    if (it == range_to_state_id_.end()) {
        TT_FATAL(false, "Range not found in SubmeshManager. Range must be provided during construction.");
    }
    return it->second;
}

}  // namespace tt::tt_metal::distributed
