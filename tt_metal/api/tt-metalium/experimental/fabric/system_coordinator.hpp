// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// SystemCoordinator: domain-level cross-host coordination for fabric bring-up.
//
// Motivation (Option B2-i)
// ------------------------
// Today every cross-host step in the fabric-configuration path is expressed as
// raw MPI-style collectives on a DistributedContext (send/recv/broadcast/
// all_gather/barrier + communicator splitting). Each of those steps actually
// follows a single semantic pattern:
//
//     serialize a domain object -> exchange it -> merge into a global view -> barrier
//
// and exchanges protobuf blobs (PhysicalSystemDescriptor, RouterPortDirectionsData,
// routing tables). SystemCoordinator captures exactly that pattern at the domain
// level so it can be backed by EITHER:
//   * CollectiveCoordinator  - wraps the existing DistributedContext (MPI or
//                              single-host); behaviour-preserving, used by the
//                              inline/workload path.
//   * ServiceCoordinator     - talks to the fabric-manager controller over gRPC;
//                              carries NO MPI dependency, used by the fabric
//                              manager service (controller + agents).
//
// This deliberately draws the abstraction boundary at the fabric-coordination
// domain rather than at the MPI-verb level, so the service backend never has to
// emulate tags, general point-to-point, or communicator splitting.
//
// This is a PUBLIC API header on purpose: it lets the concrete transport backend
// (e.g. a gRPC ServiceCoordinator) be implemented ENTIRELY outside tt_metal -- in
// the fabric-manager tool/service -- while tt_metal only depends on this
// interface. An external backend implements only the transport methods
// (bytes-in/bytes-out); the merge logic stays in tt_metal via apply_merge(),
// which reduce() invokes agent-side. See ControlPlane::set_system_coordinator.
//

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>  // tt::tt_fabric::MeshId

namespace tt::tt_fabric::coordination {

using Bytes = std::vector<uint8_t>;

// A participant group for an operation.
//   * mesh_id == nullopt  -> the whole system (all agents / world)
//   * mesh_id == set       -> only the agents that own host-ranks of that mesh
// Scope replaces DistributedContext sub-communicators (create_sub_context/split);
// the controller already knows mesh membership, so scoping is just a parameter.
struct Scope {
    std::optional<tt::tt_fabric::MeshId> mesh_id = std::nullopt;

    static Scope world() { return Scope{}; }
    static Scope mesh(tt::tt_fabric::MeshId id) { return Scope{id}; }
    bool is_world() const { return !mesh_id.has_value(); }
};

// Identifies a domain merge operation.
//
// Kept as an enum (not a raw std::function) on purpose: a central,
// controller-side implementation can dispatch it by value, while a distributed
// implementation runs the identical logic locally via apply_merge(). Central
// execution is therefore an OPTIMIZATION, never a requirement -- see reduce().
enum class MergeOp : uint8_t {
    PhysicalSystemDescriptor,  // PhysicalSystemDescriptor::merge (+ fw-version validation)
    RouterPortDirections,      // union-merge of router_port_directions maps
    // extend as further exchange points are migrated
};

// Pure merge dispatch over serialized contributions. Lives in tt_metal (shared),
// so it can run agent-side (default reduce()) or be linked by the controller for
// central merge. Nothing in the interface depends on WHERE this runs.
Bytes apply_merge(MergeOp op, const std::vector<Bytes>& contributions);

class SystemCoordinator {
public:
    virtual ~SystemCoordinator() = default;

    // --- Topology --------------------------------------------------------
    // Replaces the `distributed_context.size() == 1` / `rank()` checks scattered
    // through the control plane. IMPORTANT: an agent built without MPI has a
    // single-host DistributedContext (size()==1); correctness relies on the
    // control plane keying its short-circuits on is_distributed()/local_index()
    // here rather than on DistributedContext::size().
    [[nodiscard]] virtual bool is_distributed() const = 0;
    [[nodiscard]] virtual int participant_count(const Scope& scope) const = 0;
    [[nodiscard]] virtual int local_index(const Scope& scope) const = 0;  // 0-based within scope
    [[nodiscard]] bool is_coordinator(const Scope& scope) const { return local_index(scope) == 0; }

    // --- Phase synchronization ------------------------------------------
    virtual void barrier(const Scope& scope) = 0;

    // --- Load-bearing transport primitive --------------------------------
    // Every participant contributes `local`; every participant receives all
    // contributions in participant-index order. Merge is the CALLER's job.
    // Both backends MUST implement this; it is the primitive reduce() falls back
    // to, guaranteeing central merge is never assumed.
    [[nodiscard]] virtual std::vector<Bytes> all_gather(const Bytes& local, const Scope& scope) = 0;

    // One-to-many distribution from the participant at `root_index`.
    [[nodiscard]] virtual Bytes broadcast(const Bytes& value, int root_index, const Scope& scope) = 0;

    // --- Convenience: contribute + receive merged global view ------------
    // Default implementation = all_gather + local apply_merge, so ANY backend is
    // correct without a central merger. The ServiceCoordinator MAY override this
    // to have the controller merge once and hand back an O(1)-per-agent result
    // (sensible at current fabric sizes) -- but that remains an optimization, not
    // a contract.
    [[nodiscard]] virtual Bytes reduce(const Bytes& local, MergeOp op, const Scope& scope) {
        if (!is_distributed() || participant_count(scope) <= 1) {
            return local;
        }
        return apply_merge(op, all_gather(local, scope));
    }
};

}  // namespace tt::tt_fabric::coordination
