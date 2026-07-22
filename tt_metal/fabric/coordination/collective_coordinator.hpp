// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// CollectiveCoordinator: SystemCoordinator backed by the existing DistributedContext
// (MPI in a distributed build, SingleHostContext otherwise).
//
// This is intentionally a THIN, behaviour-preserving wrapper: each method
// reproduces the exact serialize/exchange/merge/barrier idioms already present in
// control_plane.cpp / physical_system_discovery.cpp. It exists so that:
//   (1) the inline/workload path can eventually go through SystemCoordinator with
//       zero behavioural change (step 2 "soak" -- removes the guard duplication), and
//   (2) the abstraction is proven to capture current behaviour before the gRPC
//       backend is introduced.
//
// It is NOT wired into the control plane in step 1 (workload path stays untouched).
//

#include <memory>
#include <unordered_map>

#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/system_coordinator.hpp>

namespace tt::tt_fabric::coordination {

class CollectiveCoordinator final : public SystemCoordinator {
public:
    using DistributedContext = tt::tt_metal::distributed::multihost::DistributedContext;

    // `world` is typically MetalContext::global_distributed_context(). Per-mesh
    // sub-contexts are supplied via register_mesh_context() (ControlPlane already
    // builds them in initialize_distributed_contexts()); mesh-scoped ops then
    // resolve to the registered sub-context.
    explicit CollectiveCoordinator(std::shared_ptr<DistributedContext> world);

    // Register the DistributedContext (sub-communicator) backing a mesh scope.
    // Called by ControlPlane::initialize_distributed_contexts once the per-mesh
    // sub-contexts exist. Re-registration overwrites.
    void register_mesh_context(tt::tt_fabric::MeshId mesh_id, std::shared_ptr<DistributedContext> ctx);

    [[nodiscard]] bool is_distributed() const override;
    [[nodiscard]] int participant_count(const Scope& scope) const override;
    [[nodiscard]] int local_index(const Scope& scope) const override;

    void barrier(const Scope& scope) override;

    // all_gather is implemented with the existing variable-length round-robin
    // broadcast idiom (see collect_and_merge_router_port_directions_from_all_hosts):
    // for each root in scope: broadcast(size) + broadcast(payload); everyone keeps
    // the ordered contributions. Portable across MPI and handles variable length.
    [[nodiscard]] std::vector<Bytes> all_gather(const Bytes& local, const Scope& scope) override;

    [[nodiscard]] Bytes broadcast(const Bytes& value, int root_index, const Scope& scope) override;

    // reduce() uses the base-class default (all_gather + apply_merge). The
    // collective backend has no central merger, which is exactly the property that
    // keeps central merge non-load-bearing.

private:
    // Resolves the DistributedContext for a scope. World -> world_. Mesh -> the
    // registered per-mesh sub-context.
    const DistributedContext& context_for(const Scope& scope) const;

    std::shared_ptr<DistributedContext> world_;
    std::unordered_map<tt::tt_fabric::MeshId, std::shared_ptr<DistributedContext>> mesh_contexts_;
};

}  // namespace tt::tt_fabric::coordination
