// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// ServiceCoordinator: a SystemCoordinator backed by the fabric-manager controller,
// carrying NO MPI dependency. It lives entirely in the tool (Option (a)): tt_metal
// only sees the public SystemCoordinator interface, and the transport (in-process /
// TCP / future gRPC) is injected here.
//
// Topology: controller + agents. Each agent process constructs a ServiceCoordinator
// with (a) its identity/membership (world index+size and, per mesh, its index+size),
// which the fabric-manager service already computes and passes in via args/env, and
// (b) a ControllerTransport. local_index()==0 within a scope is the coordinator role
// for that scope.
//
// Merge stays agent-side: reduce() is NOT overridden, so the base class default
// (all_gather + apply_merge, with apply_merge living in tt_metal) runs on each
// agent. The controller therefore never links fabric/domain code -- it is a dumb
// relay. Central merge remains an optional future optimization.
//

#include <cstdint>
#include <map>
#include <memory>
#include <optional>

#include <tt-metalium/experimental/fabric/system_coordinator.hpp>

#include "tools/scaleout/fabric_manager/coordination/transport.hpp"

namespace tt::scaleout_tools::fabric_manager {

// Per-mesh membership for this agent (index within the mesh + participant count).
struct MeshMembership {
    int index = 0;
    int count = 1;
};

// Identity + membership handed to the agent by the fabric-manager service (which
// already has an agent registry and computes ranks/mesh ids/mesh host ranks).
struct AgentIdentity {
    int world_index = 0;  // this agent's global participant index
    int world_size = 1;   // total number of agents
    // mesh_id -> this agent's (index, count) within that mesh. Meshes this agent
    // does not participate in are simply absent.
    std::map<uint32_t, MeshMembership> mesh_membership;
};

class ServiceCoordinator final : public tt::tt_fabric::coordination::SystemCoordinator {
public:
    using SystemCoordinator = tt::tt_fabric::coordination::SystemCoordinator;
    using Scope = tt::tt_fabric::coordination::Scope;
    using Bytes = tt::tt_fabric::coordination::Bytes;

    ServiceCoordinator(AgentIdentity identity, std::shared_ptr<ControllerTransport> transport);

    [[nodiscard]] bool is_distributed() const override { return identity_.world_size > 1; }
    [[nodiscard]] int participant_count(const Scope& scope) const override;
    [[nodiscard]] int local_index(const Scope& scope) const override;

    void barrier(const Scope& scope) override;
    [[nodiscard]] std::vector<Bytes> all_gather(const Bytes& local, const Scope& scope) override;
    [[nodiscard]] Bytes broadcast(const Bytes& value, int root_index, const Scope& scope) override;
    // reduce() intentionally NOT overridden: base-class all_gather + apply_merge.

private:
    static ScopeKey to_key(const Scope& scope);
    const MeshMembership& membership_for(const Scope& scope) const;
    uint64_t next_epoch(const Scope& scope);  // per-scope phase counter for rendezvous matching

    AgentIdentity identity_;
    std::shared_ptr<ControllerTransport> transport_;
    std::map<std::optional<uint32_t>, uint64_t> epochs_;  // scope key -> next epoch
};

}  // namespace tt::scaleout_tools::fabric_manager
