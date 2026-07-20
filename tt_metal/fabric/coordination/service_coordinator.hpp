// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// ServiceCoordinator: SystemCoordinator backed by the fabric-manager controller
// over gRPC. Carries NO MPI dependency.
//
// Topology: controller + agents. Each agent process constructs a ServiceCoordinator
// pointing at the controller endpoint, using the (rank, mesh_id, mesh_host_rank)
// bindings the fabric-manager service already computes (supplied via args/env, see
// AgentIdentity). local_index()==0 within a scope corresponds to the coordinator
// role for that scope.
//
// Mapping onto the wire contract (schemas/fabric_coordinator.proto):
//   barrier(scope)          -> Barrier RPC (controller rendezvous)
//   all_gather(local,scope) -> AllGather RPC (report local, receive all in order)
//   broadcast(v,root,scope) -> Broadcast RPC
//   reduce(local,op,scope)  -> Reduce RPC   (controller merges centrally; OPTIONAL
//                              optimization -- falls back to base-class all_gather +
//                              apply_merge if a controller reports merge_unsupported)
//
// Every RPC carries a monotonically increasing epoch/phase id per scope so the
// controller can match participants of the same collective without MPI-style tags.
//

#include <cstdint>
#include <memory>
#include <string>

#include <tt-metalium/experimental/fabric/system_coordinator.hpp>

namespace tt::tt_fabric::coordination {

// Identity + membership handed to the agent by the fabric-manager service.
// (The service already has an agent registry and computes ranks/mesh ids/mesh
// host ranks; the PoC just plumbs those in.)
struct AgentIdentity {
    std::string controller_endpoint;  // host:port of the controller gRPC server
    int world_index = 0;              // this agent's global participant index
    int world_size = 1;               // total number of agents
    // mesh membership (mesh_id -> this agent's index within that mesh, and the
    // mesh's participant count) is fetched at Register() and cached; omitted here.
};

class ServiceCoordinator final : public SystemCoordinator {
public:
    explicit ServiceCoordinator(AgentIdentity identity);
    ~ServiceCoordinator() override;

    [[nodiscard]] bool is_distributed() const override { return identity_.world_size > 1; }
    [[nodiscard]] int participant_count(const Scope& scope) const override;
    [[nodiscard]] int local_index(const Scope& scope) const override;

    void barrier(const Scope& scope) override;
    [[nodiscard]] std::vector<Bytes> all_gather(const Bytes& local, const Scope& scope) override;
    [[nodiscard]] Bytes broadcast(const Bytes& value, int root_index, const Scope& scope) override;

    // Central-merge optimization: report local -> controller applies apply_merge()
    // -> fetch merged. If the controller cannot merge (e.g. no merge lib linked),
    // it replies merge_unsupported and we defer to SystemCoordinator::reduce.
    [[nodiscard]] Bytes reduce(const Bytes& local, MergeOp op, const Scope& scope) override;

private:
    uint64_t next_epoch(const Scope& scope);  // per-scope phase counter for RPC matching

    AgentIdentity identity_;
    struct Impl;  // hides the gRPC stub / channel from this header
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_fabric::coordination
