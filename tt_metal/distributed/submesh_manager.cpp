// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "submesh_manager.hpp"

namespace tt::tt_metal::distributed {

SubmeshState::Manager::Manager(const std::vector<MeshCoordinateRange>& ranges) {
    for (size_t i = 0; i < ranges.size(); ++i) {
        StateID state_id = static_cast<StateID>(i);
        range_to_state_id_[ranges[i]] = state_id;
    }

    // Build dependency map based on overlap criteria
    std::unordered_map<AllocatorID, tt::stl::SmallVector<AllocatorID>> dependencies_map;

    for (size_t i = 0; i < ranges.size(); ++i) {
        AllocatorID allocator_i{static_cast<uint32_t>(i)};
        tt::stl::SmallVector<AllocatorID> dependencies;

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

const SubmeshState::Manager::AllocatorDependencies& SubmeshState::Manager::get_dependencies() const {
    return dependencies_;
}

SubmeshState::StateID SubmeshState::Manager::get_state_id(const MeshCoordinateRange& range) const {
    auto it = range_to_state_id_.find(range);
    TT_FATAL(it != range_to_state_id_.end(), "Range not found in SubmeshState::Manager.");
    return it->second;
}

std::vector<std::shared_ptr<SubmeshState>> SubmeshState::create_states_for_ranges(
    const std::vector<MeshCoordinateRange>& ranges) {
    auto shared_manager = std::make_shared<Manager>(ranges);
    std::vector<std::shared_ptr<SubmeshState>> states;
    states.reserve(ranges.size());
    for (const auto& range : ranges) {
        states.push_back(std::make_shared<SubmeshState>(shared_manager, shared_manager->get_state_id(range)));
    }
    return states;
}

}  // namespace tt::tt_metal::distributed
