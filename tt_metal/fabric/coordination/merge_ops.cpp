// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/system_coordinator.hpp>

#include <algorithm>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"
#include "tt_metal/fabric/serialization/router_port_directions.hpp"

//
// apply_merge: pure, shared merge dispatch over serialized contributions.
//
// This is deliberately transport-agnostic and location-agnostic. It is used as
// the agent-side fallback by SystemCoordinator::reduce() and MAY also be linked by
// the controller for the central-merge optimization. Keeping it here (shared
// tt_metal) is what lets central merge stay an optimization rather than a
// contract: the same bytes-in/bytes-out function runs correctly in either place.
//
// The merge logic mirrors the existing control-plane idioms:
//   * PhysicalSystemDescriptor: PhysicalSystemDescriptor::merge (see
//     physical_system_discovery.cpp exchange_metadata()).
//   * RouterPortDirections: union of router_port_directions maps (see
//     control_plane.cpp collect_and_merge_router_port_directions_from_all_hosts()).
//
// NOTE (draft): the firmware-version validation that exchange_metadata() performs
// during PSD merge is intentionally omitted here; it should be re-introduced (as a
// pure check) when this replaces the inline path in the step-2 soak.
//

namespace tt::tt_fabric::coordination {

namespace {

Bytes merge_physical_system_descriptors(const std::vector<Bytes>& contributions) {
    using tt::tt_metal::deserialize_physical_system_descriptor_from_bytes;
    using tt::tt_metal::serialize_physical_system_descriptor_to_bytes;

    auto merged = deserialize_physical_system_descriptor_from_bytes(contributions.front());
    for (std::size_t i = 1; i < contributions.size(); ++i) {
        auto peer = deserialize_physical_system_descriptor_from_bytes(contributions[i]);
        merged.merge(std::move(peer));
    }
    return serialize_physical_system_descriptor_to_bytes(merged);
}

Bytes merge_router_port_directions(const std::vector<Bytes>& contributions) {
    auto merged = deserialize_router_port_directions_from_bytes(contributions.front());
    for (std::size_t i = 1; i < contributions.size(); ++i) {
        auto remote = deserialize_router_port_directions_from_bytes(contributions[i]);
        for (const auto& [fabric_node_id, direction_map] : remote.router_port_directions_map) {
            auto [it, inserted] = merged.router_port_directions_map.try_emplace(fabric_node_id, direction_map);
            if (inserted) {
                continue;
            }
            auto& local_direction_map = it->second;
            for (const auto& [direction, channels] : direction_map) {
                auto [dir_it, dir_inserted] = local_direction_map.try_emplace(direction, channels);
                if (dir_inserted) {
                    continue;
                }
                auto& local_channels = dir_it->second;
                for (const auto& channel : channels) {
                    if (std::find(local_channels.begin(), local_channels.end(), channel) == local_channels.end()) {
                        local_channels.push_back(channel);
                    }
                }
            }
        }
    }
    return serialize_router_port_directions_to_bytes(merged);
}

}  // namespace

Bytes apply_merge(MergeOp op, const std::vector<Bytes>& contributions) {
    TT_FATAL(!contributions.empty(), "apply_merge requires at least one contribution");
    if (contributions.size() == 1) {
        return contributions.front();
    }
    switch (op) {
        case MergeOp::PhysicalSystemDescriptor: return merge_physical_system_descriptors(contributions);
        case MergeOp::RouterPortDirections: return merge_router_port_directions(contributions);
    }
    TT_THROW("apply_merge: unhandled MergeOp");
}

}  // namespace tt::tt_fabric::coordination
