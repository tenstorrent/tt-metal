// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// ControllerTransport: the single wire primitive the ServiceCoordinator needs.
//
// Option (a) draws the transport boundary as narrow as possible: every fabric
// coordination step (barrier / all_gather / broadcast) reduces to ONE rendezvous
// collective -- "everyone in a scope contributes a byte blob for a given phase,
// everyone receives all contributions in participant-index order". barrier is
// that with empty payloads; broadcast is that where the caller keeps only the
// root's slot. Because each SystemCoordinator call bumps a per-scope epoch, the
// (scope, epoch) pair uniquely identifies a collective instance without MPI-style
// tags -- so the controller needs no knowledge of op types.
//
// This interface is deliberately transport-agnostic: InProcessTransport calls a
// shared Controller directly (for local/self-test), TcpTransport carries it over
// sockets (cross-process / cross-host), and a future gRPC transport is a drop-in
// implementing the same one method. None of these leak into tt_metal.
//

#include <cstdint>
#include <optional>
#include <vector>

namespace tt::scaleout_tools::fabric_manager {

using Bytes = std::vector<uint8_t>;

// A participant group. nullopt => the whole system (world); a value => the agents
// that own host-ranks of that mesh id. Mirrors tt_fabric::coordination::Scope but
// keeps the transport free of tt_metal domain types.
struct ScopeKey {
    std::optional<uint32_t> mesh_id;

    bool operator==(const ScopeKey& other) const { return mesh_id == other.mesh_id; }
};

class ControllerTransport {
public:
    virtual ~ControllerTransport() = default;

    // Contribute `payload` as participant `index` of `count` for phase `epoch` of
    // `scope`; block until all `count` participants have contributed, then return
    // every contribution in ascending participant-index order.
    [[nodiscard]] virtual std::vector<Bytes> exchange(
        const ScopeKey& scope, uint64_t epoch, int index, int count, const Bytes& payload) = 0;
};

}  // namespace tt::scaleout_tools::fabric_manager
