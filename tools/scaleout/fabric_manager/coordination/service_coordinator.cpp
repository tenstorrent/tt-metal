// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/fabric_manager/coordination/service_coordinator.hpp"

#include <utility>

#include <tt_stl/assert.hpp>

namespace tt::scaleout_tools::fabric_manager {

ServiceCoordinator::ServiceCoordinator(AgentIdentity identity, std::shared_ptr<ControllerTransport> transport) :
    identity_(std::move(identity)), transport_(std::move(transport)) {
    TT_FATAL(transport_ != nullptr, "ServiceCoordinator requires a non-null transport");
    TT_FATAL(identity_.world_size >= 1, "ServiceCoordinator world_size must be >= 1");
    TT_FATAL(
        identity_.world_index >= 0 && identity_.world_index < identity_.world_size,
        "ServiceCoordinator world_index {} out of range [0,{})",
        identity_.world_index,
        identity_.world_size);
}

ScopeKey ServiceCoordinator::to_key(const Scope& scope) {
    if (scope.is_world()) {
        return ScopeKey{std::nullopt};
    }
    return ScopeKey{static_cast<uint32_t>(*(scope.mesh_id.value()))};
}

const MeshMembership& ServiceCoordinator::membership_for(const Scope& scope) const {
    // World scope membership is synthesized from the agent's world identity; keep it
    // static-thread-local so we can return a reference uniformly with mesh scopes.
    static thread_local MeshMembership world_membership;
    if (scope.is_world()) {
        world_membership = MeshMembership{identity_.world_index, identity_.world_size};
        return world_membership;
    }
    auto mesh_id = static_cast<uint32_t>(*(scope.mesh_id.value()));
    auto it = identity_.mesh_membership.find(mesh_id);
    TT_FATAL(
        it != identity_.mesh_membership.end(),
        "ServiceCoordinator: agent has no membership for mesh {} (supply it in AgentIdentity)",
        mesh_id);
    return it->second;
}

int ServiceCoordinator::participant_count(const Scope& scope) const { return membership_for(scope).count; }

int ServiceCoordinator::local_index(const Scope& scope) const { return membership_for(scope).index; }

uint64_t ServiceCoordinator::next_epoch(const Scope& scope) {
    auto key = to_key(scope).mesh_id;
    return epochs_[key]++;
}

void ServiceCoordinator::barrier(const Scope& scope) {
    const auto& m = membership_for(scope);
    if (m.count <= 1) {
        return;
    }
    (void)transport_->exchange(to_key(scope), next_epoch(scope), m.index, m.count, Bytes{});
}

std::vector<ServiceCoordinator::Bytes> ServiceCoordinator::all_gather(const Bytes& local, const Scope& scope) {
    const auto& m = membership_for(scope);
    if (m.count <= 1) {
        return {local};
    }
    return transport_->exchange(to_key(scope), next_epoch(scope), m.index, m.count, local);
}

ServiceCoordinator::Bytes ServiceCoordinator::broadcast(const Bytes& value, int root_index, const Scope& scope) {
    const auto& m = membership_for(scope);
    if (m.count <= 1) {
        return value;
    }
    // Only the root's payload matters; everyone else contributes empty. All receive the
    // full gathered set and keep the root's slot.
    const Bytes& contribution = (m.index == root_index) ? value : Bytes{};
    auto gathered = transport_->exchange(to_key(scope), next_epoch(scope), m.index, m.count, contribution);
    TT_FATAL(
        root_index >= 0 && root_index < static_cast<int>(gathered.size()),
        "ServiceCoordinator::broadcast root_index {} out of range [0,{})",
        root_index,
        gathered.size());
    return gathered[static_cast<std::size_t>(root_index)];
}

}  // namespace tt::scaleout_tools::fabric_manager
