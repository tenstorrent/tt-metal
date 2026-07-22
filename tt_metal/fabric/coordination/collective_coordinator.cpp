// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/coordination/collective_coordinator.hpp"

#include <cstring>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_fabric::coordination {

using tt::tt_metal::distributed::multihost::Rank;

CollectiveCoordinator::CollectiveCoordinator(std::shared_ptr<DistributedContext> world) : world_(std::move(world)) {
    TT_FATAL(world_ != nullptr, "CollectiveCoordinator requires a non-null world context");
}

void CollectiveCoordinator::register_mesh_context(
    tt::tt_fabric::MeshId mesh_id, std::shared_ptr<DistributedContext> ctx) {
    TT_FATAL(ctx != nullptr, "register_mesh_context requires a non-null context for mesh {}", *mesh_id);
    mesh_contexts_[mesh_id] = std::move(ctx);
}

const CollectiveCoordinator::DistributedContext& CollectiveCoordinator::context_for(const Scope& scope) const {
    if (scope.is_world()) {
        return *world_;
    }
    auto it = mesh_contexts_.find(*scope.mesh_id);
    TT_FATAL(
        it != mesh_contexts_.end(),
        "CollectiveCoordinator: no registered context for mesh {} (call register_mesh_context first)",
        **scope.mesh_id);
    return *it->second;
}

bool CollectiveCoordinator::is_distributed() const { return *world_->size() > 1; }

int CollectiveCoordinator::participant_count(const Scope& scope) const { return *context_for(scope).size(); }

int CollectiveCoordinator::local_index(const Scope& scope) const { return *context_for(scope).rank(); }

void CollectiveCoordinator::barrier(const Scope& scope) { context_for(scope).barrier(); }

std::vector<Bytes> CollectiveCoordinator::all_gather(const Bytes& local, const Scope& scope) {
    const auto& ctx = context_for(scope);
    const int n = *ctx.size();
    const int me = *ctx.rank();

    std::vector<Bytes> result(static_cast<std::size_t>(n));

    // Variable-length all-gather via round-robin broadcast (size then payload). This mirrors the
    // existing idiom in collect_and_merge_router_port_directions_from_all_hosts() and is portable
    // across MPI and single-host builds.
    for (int root = 0; root < n; ++root) {
        std::uint64_t payload_size = 0;
        if (root == me) {
            result[static_cast<std::size_t>(root)] = local;
            payload_size = local.size();
        }

        ctx.broadcast(
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&payload_size), sizeof(payload_size)), Rank{root});

        auto& slot = result[static_cast<std::size_t>(root)];
        if (root != me) {
            slot.resize(payload_size);
        }
        if (payload_size > 0) {
            ctx.broadcast(ttsl::as_writable_bytes(ttsl::Span<std::uint8_t>(slot.data(), slot.size())), Rank{root});
        }
    }
    return result;
}

Bytes CollectiveCoordinator::broadcast(const Bytes& value, int root_index, const Scope& scope) {
    const auto& ctx = context_for(scope);
    const int me = *ctx.rank();

    Bytes buffer;
    std::uint64_t payload_size = 0;
    if (me == root_index) {
        buffer = value;
        payload_size = buffer.size();
    }

    ctx.broadcast(
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&payload_size), sizeof(payload_size)), Rank{root_index});

    if (me != root_index) {
        buffer.resize(payload_size);
    }
    if (payload_size > 0) {
        ctx.broadcast(
            ttsl::as_writable_bytes(ttsl::Span<std::uint8_t>(buffer.data(), buffer.size())), Rank{root_index});
    }
    return buffer;
}

}  // namespace tt::tt_fabric::coordination
